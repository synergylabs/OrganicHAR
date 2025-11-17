import warnings
from typing import List, Dict, Any, Optional
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import tqdm.auto as tqdm

warnings.filterwarnings('ignore', category=FutureWarning, message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")

# Import dataclasses from __init__.py
from . import FeatureResults, ClusteringResults

class SensorClusteringConfig:
    """Configuration for sensor-specific clustering parameters"""
    n_components_range: List[int] = None
    n_clusters_range: List[int] = None
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    def __post_init__(self):
        # Set defaults if not provided
        if self.n_components_range is None:
            self.n_components_range = [8, 12, 15, 20, 40, 80]
        if self.n_clusters_range is None:
            self.n_clusters_range = [5, 8, 10, 15, 20, 30, 40]

class KMeansClustering:
    def __init__(self, write_dir: str, config: SensorClusteringConfig = None):
        self.write_dir = write_dir
        os.makedirs(write_dir, exist_ok=True)
        self.config = config if config else SensorClusteringConfig()
        self.scaler = RobustScaler()
        self.power_transformer = PowerTransformer()
        self.pca = None
        self.kmeans = None
        self.cluster_centroids = None
        self.feature_mask = None

    def fit(self, feature_results: FeatureResults) -> ClusteringResults:
        if len(feature_results.window_ids) < 10:
            raise ValueError("Need at least 10 windows for clustering")
        features = feature_results.features
        feature_names = np.array(feature_results.feature_names)
        window_ids = feature_results.window_ids
        features = np.nan_to_num(features, nan=0.0)
        variances = np.var(features, axis=0)
        q75, q25 = np.percentile(features, [75, 25], axis=0)
        iqr = q75 - q25
        stable_feature_mask = (variances > 1e-6) & (iqr > 1e-6)
        self.feature_mask = stable_feature_mask
        features = features[:, self.feature_mask]
        feature_names = feature_names[self.feature_mask]
        if features.shape[1] == 0:
            raise ValueError("No numerically stable features found. Clustering cannot proceed.")
        features_scaled = self.scaler.fit_transform(features)
        features_transformed = self.power_transformer.fit_transform(features_scaled)
        features_transformed = np.nan_to_num(features_transformed, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        features_transformed = np.clip(features_transformed, np.finfo(np.float32).min, np.finfo(np.float32).max)
        best_params = self._find_best_parameters(features_transformed)
        if best_params is None:
            raise RuntimeError("Could not find suitable clustering parameters")
        if best_params['n_components'] >= features_transformed.shape[1]:
            features_reduced = features_transformed
        else:
            self.pca = PCA(n_components=best_params['n_components'])
            features_reduced = self.pca.fit_transform(features_transformed)
        self.kmeans = KMeans(n_clusters=best_params['n_clusters'], random_state=42, n_init='auto')
        labels = self.kmeans.fit_predict(features_reduced)
        self._calculate_cluster_centroids(features_reduced, labels)
        # For KMeans, confidence is not defined; use silhouette score for each point if desired, else set to 1.0
        try:
            if len(set(labels)) > 1:
                confidences = silhouette_score(features_reduced, labels) * np.ones_like(labels, dtype=float)
            else:
                confidences = np.zeros_like(labels, dtype=float)
        except Exception:
            confidences = np.zeros_like(labels, dtype=float)
        return ClusteringResults(
            window_ids=window_ids,
            labels=labels,
            features=features,
            confidences=confidences,
            feature_names=feature_names.tolist()
        )

    def _find_best_parameters(self, features_transformed: np.ndarray) -> Optional[Dict[str, Any]]:
        best_score = -np.inf
        best_params = None
        for n_components in tqdm.tqdm(self.config.n_components_range, desc="Grid Search: n_components"):
            if n_components >= features_transformed.shape[1]:
                features_reduced = features_transformed
            else:
                pca = PCA(n_components=n_components)
                features_reduced = pca.fit_transform(features_transformed)
            for n_clusters in tqdm.tqdm(getattr(self.config, 'n_clusters_range', [5, 8, 10, 15, 20, 30, 40]), desc="Grid Search: n_clusters"):
                if n_clusters >= len(features_reduced):
                    continue
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                    labels = kmeans.fit_predict(features_reduced)
                    if len(set(labels)) < 2:
                        continue
                    sil_score = silhouette_score(features_reduced, labels)
                    db_score = -davies_bouldin_score(features_reduced, labels)  # lower is better, so negate
                    ch_score = calinski_harabasz_score(features_reduced, labels)
                    # Composite score: prioritize silhouette, then ch_score, penalize db_score
                    score = 0.6 * sil_score + 0.2 * (ch_score / 1000) + 0.2 * (db_score / 10)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'n_components': n_components,
                            'n_clusters': n_clusters
                        }
                except Exception:
                    continue
        return best_params

    def _calculate_cluster_centroids(self, features: np.ndarray, labels: np.ndarray) -> None:
        unique_labels = np.unique(labels)
        self.cluster_centroids = []
        for label in unique_labels:
            mask = labels == label
            centroid = np.mean(features[mask], axis=0)
            self.cluster_centroids.append(centroid)
        self.cluster_centroids = np.array(self.cluster_centroids)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.kmeans is None or self.feature_mask is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        features = features[:, self.feature_mask]
        features_scaled = self.scaler.transform(features)
        features_transformed = self.power_transformer.transform(features_scaled)
        features_transformed = np.nan_to_num(features_transformed, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        features_transformed = np.clip(features_transformed, np.finfo(np.float32).min, np.finfo(np.float32).max)
        if self.pca:
            features_reduced = self.pca.transform(features_transformed)
        else:
            features_reduced = features_transformed
        labels = self.kmeans.predict(features_reduced)
        return labels

    def save(self, path: str):
        if self.kmeans is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        model_data = {
            'scaler': self.scaler,
            'power_transformer': self.power_transformer,
            'pca': self.pca,
            'kmeans': self.kmeans,
            'config': self.config,
            'cluster_centroids': self.cluster_centroids,
            'feature_mask': self.feature_mask
        }
        joblib.dump(model_data, path)

    @staticmethod
    def load(path: str, write_dir: str) -> 'KMeansClustering':
        model_data = joblib.load(path)
        instance = KMeansClustering(write_dir=write_dir, config=model_data['config'])
        instance.scaler = model_data['scaler']
        instance.power_transformer = model_data.get('power_transformer')
        instance.pca = model_data['pca']
        instance.kmeans = model_data['kmeans']
        instance.cluster_centroids = model_data.get('cluster_centroids')
        instance.feature_mask = model_data.get('feature_mask')
        return instance
