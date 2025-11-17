import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import os
import hdbscan
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
import joblib
import tqdm.auto as tqdm
from scipy.stats import describe

warnings.filterwarnings('ignore', category=FutureWarning, message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")

# Import dataclasses from __init__.py
from . import FeatureResults, ClusteringResults

@dataclass
class SensorClusteringConfig:
    """Configuration for sensor-specific clustering parameters"""
    # PCA parameters
    n_components_range: List[int] = None
    
    # HDBSCAN parameters
    min_cluster_size_range: List[int] = None
    min_samples_range: List[int] = None
    
    # Evaluation parameters
    min_desired_clusters: int = 20
    max_desired_clusters: int = 40
    max_noise_ratio: float = 0.8
    min_cluster_size: int = 5
    
    def __post_init__(self):
        # Set defaults if not provided
        if self.n_components_range is None:
            self.n_components_range = [8, 12, 15, 20, 40, 80]
        if self.min_cluster_size_range is None:
            self.min_cluster_size_range = [5, 8, 10, 15]
        if self.min_samples_range is None:
            self.min_samples_range = [3, 5, 8]


class HDBSCANClustering:
    def __init__(self, write_dir: str, config: SensorClusteringConfig = None):
        """
        Initialize HDBSCAN clustering with output directory and configuration.
        
        Args:
            write_dir: Directory to write outputs
            config: Sensor-specific clustering configuration
        """
        self.write_dir = write_dir
        os.makedirs(write_dir, exist_ok=True)
        
        # Use default config if not provided
        self.config = config if config else SensorClusteringConfig()
        
        # Initialize components
        self.scaler = RobustScaler()
        self.power_transformer = PowerTransformer()
        self.pca = None
        self.clustering = None
        self.cluster_centroids = None
        self.feature_mask = None

    def fit(self, feature_results: FeatureResults) -> ClusteringResults:
        """
        Fit clustering model to features.
        
        Args:
            feature_results: Features extracted from sensor data
            
        Returns:
            ClusteringResults containing cluster assignments and metadata
        """
        # Check if we have enough data
        if len(feature_results.window_ids) < 10:
            raise ValueError("Need at least 10 windows for clustering")
            
        # Get feature data
        features = feature_results.features
        feature_names = np.array(feature_results.feature_names)
        window_ids = feature_results.window_ids
        
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)

        # Stricter pre-flight check for numerical stability. A feature is only kept if
        # it has BOTH sufficient variance AND a sufficient Interquartile Range (IQR)
        # to prevent RobustScaler from creating extreme or infinite values.
        variances = np.var(features, axis=0)
        q75, q25 = np.percentile(features, [75, 25], axis=0)
        iqr = q75 - q25
        
        # A feature is stable if both variance and IQR are above a small threshold
        stable_feature_mask = (variances > 1e-6) & (iqr > 1e-6)
        self.feature_mask = stable_feature_mask
        
        features = features[:, self.feature_mask]
        feature_names = feature_names[self.feature_mask]

        if features.shape[1] == 0:
            raise ValueError("No numerically stable features found. "
                             "Clustering cannot proceed.")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # any scaled features that are nan, inf, or -inf, set to 0
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply PowerTransformer to make data more Gaussian-like and stabilize variance
        features_transformed = self.power_transformer.fit_transform(features_scaled)

        # Final guardrail: forcefully handle any remaining non-finite values
        features_transformed = np.nan_to_num(features_transformed, 
                                        nan=0.0, 
                                        posinf=np.finfo(np.float32).max, 
                                        neginf=np.finfo(np.float32).min)
        features_transformed = np.clip(features_transformed, 
                                  np.finfo(np.float32).min, 
                                  np.finfo(np.float32).max)

        # Find best parameters
        best_params = self._find_best_parameters(features_transformed)
        
        if best_params is None:
            raise RuntimeError("Could not find suitable clustering parameters")
            
        # Apply final PCA
        if best_params['n_components'] >= features_transformed.shape[1]:
            features_reduced = features_transformed
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, message=".*divide by zero encountered in matmul.*")
                warnings.filterwarnings('ignore', category=RuntimeWarning, message=".*overflow encountered in matmul.*")
                warnings.filterwarnings('ignore', category=RuntimeWarning, message=".*invalid value encountered in matmul.*")
                self.pca = PCA(n_components=best_params['n_components'])
                features_reduced = self.pca.fit_transform(features_transformed)
            
        # Perform final clustering
        self.clustering = hdbscan.HDBSCAN(
            min_cluster_size=best_params['min_cluster_size'],
            min_samples=best_params['min_samples'],
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        self.clustering.fit(features_reduced)
        
        # Calculate cluster centroids
        self._calculate_cluster_centroids(features_reduced, self.clustering.labels_)
        
        return ClusteringResults(
            window_ids=window_ids,
            labels=self.clustering.labels_,
            features=features,
            confidences=self.clustering.probabilities_,
            feature_names=feature_names.tolist()
        )
    
    def _find_best_parameters(self, features_transformed: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Find best parameters using grid search.
        
        Args:
            features_transformed: Transformed feature matrix
            
        Returns:
            Dictionary of best parameters or None if no suitable parameters found
        """
        best_score = -np.inf
        best_params = None
        
        for n_components in tqdm.tqdm(self.config.n_components_range, desc="L1 Grid Search: n_components"):
            # Skip if n_components > feature dimensions
            if n_components >= features_transformed.shape[1]:
                features_reduced = features_transformed
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning, message=".*divide by zero encountered in matmul.*")
                    warnings.filterwarnings('ignore', category=RuntimeWarning, message=".*overflow encountered in matmul.*")
                    warnings.filterwarnings('ignore', category=RuntimeWarning, message=".*invalid value encountered in matmul.*")
                    pca = PCA(n_components=n_components)
                    features_reduced = pca.fit_transform(features_transformed)
                
                for min_cluster_size in tqdm.tqdm(self.config.min_cluster_size_range, desc="L2 Grid Search: min_cluster_size"):
                    for min_samples in tqdm.tqdm(self.config.min_samples_range, desc="L3 Grid Search: min_samples"):
                        try:
                            # Create and fit HDBSCAN clusterer
                            clusterer = hdbscan.HDBSCAN(
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                cluster_selection_method='eom',
                                prediction_data=True
                            )
                            
                            clusterer.fit(features_reduced)
                            
                            # Evaluate clustering
                            score = self._evaluate_clustering(
                                features_reduced,
                                clusterer.labels_,
                                clusterer.probabilities_
                            )
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'n_components': n_components,
                                    'min_cluster_size': min_cluster_size,
                                    'min_samples': min_samples
                                }
                                
                        except Exception as e:
                            warnings.warn(f"Clustering failed with parameters {n_components}, "
                                          f"{min_cluster_size}, {min_samples}: {str(e)}")
                            continue
        
        # If no valid parameters found, use defaults
        if best_params is None:
            best_params = {
                'n_components': self.config.n_components_range[0],
                'min_cluster_size': self.config.min_cluster_size_range[0],
                'min_samples': self.config.min_samples_range[0]
            }
            
        return best_params
    
    def _evaluate_clustering(self, features: np.ndarray, 
                             labels: np.ndarray, 
                             probabilities: np.ndarray) -> float:
        """
        Evaluate clustering quality.
        
        Args:
            features: Feature matrix used for clustering
            labels: Cluster assignments
            probabilities: Cluster probabilities
            
        Returns:
            Score between 0 and 1, with higher values indicating better clustering
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        
        # 1. Cluster count score
        if n_clusters < self.config.min_desired_clusters:
            cluster_count_score = n_clusters / self.config.min_desired_clusters
        elif n_clusters > self.config.max_desired_clusters:
            cluster_count_score = 1.0 - ((n_clusters - self.config.max_desired_clusters) / 
                                        self.config.max_desired_clusters)
        else:
            cluster_count_score = 1.0
            
        # 2. Cluster size scores
        size_scores = []
        tightness_scores = []
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            
            # Size score
            if cluster_size < self.config.min_cluster_size:
                size_scores.append(0.0)
            else:
                size_scores.append(min(1.0, cluster_size / 10))
                
            # Tightness score (using average distance to centroid)
            if cluster_size >= 2:
                cluster_points = features[cluster_mask]
                centroid = np.mean(cluster_points, axis=0)
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                tightness = np.exp(-np.mean(distances))
                tightness_scores.append(tightness)
                
        size_score = np.mean(size_scores) if size_scores else 0.0
        tightness_score = np.mean(tightness_scores) if tightness_scores else 0.0
        
        # 3. Noise ratio score
        noise_ratio = np.sum(labels == -1) / len(labels)
        if noise_ratio > self.config.max_noise_ratio:
            noise_score = 1.0 - (noise_ratio - self.config.max_noise_ratio) / 0.2
            noise_score = max(0.0, noise_score)
        else:
            noise_score = 1.0
            
        # 4. Confidence score
        confidence_score = np.mean(probabilities[labels != -1]) if np.any(labels != -1) else 0.0
        
        # Combine scores with weights
        final_score = (
            0.3 * cluster_count_score +
            0.2 * size_score +
            0.2 * tightness_score + 
            0.2 * noise_score +
            0.1 * confidence_score
        )
        
        return final_score
    
    def _calculate_cluster_centroids(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Calculate centroids for each cluster.
        
        Args:
            features: Feature matrix
            labels: Cluster assignments
        """
        unique_labels = np.unique(labels)
        self.cluster_centroids = []
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            mask = labels == label
            centroid = np.mean(features[mask], axis=0)
            self.cluster_centroids.append(centroid)
            
        self.cluster_centroids = np.array(self.cluster_centroids)
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster for new data points.
        
        Args:
            features: Feature matrix for new data
            
        Returns:
            Tuple of (predicted_labels, prediction_probabilities)
        """
        if self.clustering is None or self.feature_mask is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Ensure we're using the same features the model was trained on
        features = features[:, self.feature_mask]
            
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Apply the same power transform
        features_transformed = self.power_transformer.transform(features_scaled)

        # Forcefully handle any non-finite values or overflows from scaling
        features_transformed = np.nan_to_num(features_transformed, 
                                        nan=0.0, 
                                        posinf=np.finfo(np.float32).max, 
                                        neginf=np.finfo(np.float32).min)
        features_transformed = np.clip(features_transformed, 
                                  np.finfo(np.float32).min, 
                                  np.finfo(np.float32).max)

        # Apply PCA
        if self.pca:
            features_reduced = self.pca.transform(features_transformed)
        else:
            features_reduced = features_transformed
        
        # Predict clusters
        predicted_labels, probabilities = hdbscan.approximate_predict(
            self.clustering, features_reduced)
            
        return predicted_labels, probabilities

    def save(self, path: str):
        """
        Saves the fitted model to a file.
        
        Args:
            path: Path to save the model file.
        """
        if self.clustering is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        model_data = {
            'scaler': self.scaler,
            'power_transformer': self.power_transformer,
            'pca': self.pca,
            'clustering': self.clustering,
            'config': self.config,
            'cluster_centroids': self.cluster_centroids,
            'feature_mask': self.feature_mask
        }
        
        joblib.dump(model_data, path)

    @staticmethod
    def load(path: str, write_dir: str) -> 'HDBSCANClustering':
        """
        Loads a fitted model from a file.
        
        Args:
            path: Path to the model file.
            write_dir: Directory to write any new outputs.
            
        Returns:
            A new instance of HDBSCANClustering with the loaded model.
        """
        model_data = joblib.load(path)
        
        instance = HDBSCANClustering(write_dir=write_dir, config=model_data['config'])
        
        instance.scaler = model_data['scaler']
        instance.power_transformer = model_data.get('power_transformer')
        instance.pca = model_data['pca']
        instance.clustering = model_data['clustering']
        instance.cluster_centroids = model_data.get('cluster_centroids')
        instance.feature_mask = model_data.get('feature_mask')
        
        return instance
    
