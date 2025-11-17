import os
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import json
import sys
import pickle
from tqdm.auto import tqdm
import warnings
import traceback
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from common.model_training.classifiers import get_trainer

base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")
base_results_dir = "/Volumes/Research-Prasoon/OrganicHAR/inhome_evaluation"
os.makedirs(base_results_dir, exist_ok=True)

parser = argparse.ArgumentParser(description="Train best models using sequential selection results.")
parser.add_argument("--participant", default="P5data-collection", help="Participant name")
parser.add_argument("--session_prefix", default="P11", help="Session prefix")
parser.add_argument("--window_size", default=5.0, help="Window size in seconds")
parser.add_argument("--sliding_window_length", default=0.5, help="Sliding window length in seconds")
parser.add_argument("--description_vlm_model", required=True, help="VLM model used for description generation")
parser.add_argument("--clustering_vlm_model", default="gpt-4.1", help="VLM model used for location and activity clustering")
parser.add_argument("--merging_threshold", default=0.3, help="Threshold for merging similar labels")
parser.add_argument("--alpha", required=True, help="Alpha for activity clustering")
parser.add_argument("--label_humanizer_model", default="gpt-4.1-mini", help="VLM model used for label humanization")
parser.add_argument("--direction", default="forward", choices=["forward", "backward"], help="Selection direction")
parser.add_argument("--debug", action="store_true", help="Debug mode")

args = parser.parse_args()

allowed_sensors = ["doppler", "pose_depth", "thermal_vision", "thermal_audio"]
allowed_sensors += ["persondepth_depth", "watchmotion_dominant"]

trainers = ['svm', 'random_forest', 'knn', 'xgboost','balanced_rf','balanced_bagging','easy_ensemble']

participant = args.participant
session_prefix = args.session_prefix
window_size = float(args.window_size)
sliding_window_length = float(args.sliding_window_length)
debug = args.debug

features_base_dir = f"{base_results_dir}/{participant}/features_{window_size}_{sliding_window_length}"
best_models_base_dir = f"{base_results_dir}/{participant}/best_models_v3_alpha_{args.alpha}_{window_size}_{sliding_window_length}/{args.description_vlm_model}/{args.merging_threshold}_control_{args.clustering_vlm_model}"
os.makedirs(best_models_base_dir, exist_ok=True)

def get_session_data(base_processed_dir, participant, session_prefix, window_size, sliding_window_length):
    watch_sessions_file = f"{base_processed_dir}/{participant}/watch_sessions_filtered.json"
    if not os.path.exists(watch_sessions_file):
        print(f"File {watch_sessions_file} does not exist. Exiting...")
        exit(1)
    with open(watch_sessions_file, "r") as f:
        watch_sessions = json.load(f)
    
    base_video_dir = f"{base_processed_dir}/{participant}/processed_video_data"
    base_depth_dir = f"{base_processed_dir}/{participant}/processed_depth_data"

    def video_file_exists(base_filename, video_dir, depth_dir):
        if os.path.exists(os.path.join(video_dir, base_filename)):
            return True
        if os.path.exists(os.path.join(depth_dir, base_filename)):
            return True
        return False
    
    collected_sessions = [
        {
            "session_key":f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}",
            **session_data,
        }
        for session_idx, session_data in enumerate(watch_sessions)
        if video_file_exists(f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}.mp4",base_video_dir, base_depth_dir)
    ]
    return collected_sessions

def parse_selected_features(selected_features):
    """Parse selected features to extract sensor-trainer pairs"""
    sensor_trainer_pairs = set()
    for feature in selected_features:
        if '__' in feature:
            parts = feature.split('__')
            sensor = parts[0]
            trainer = parts[1].replace('_pred', '').replace('_conf', '')
            sensor_trainer_pairs.add((sensor, trainer))
    return list(sensor_trainer_pairs)

collected_sessions = get_session_data(base_processed_dir, participant, session_prefix, window_size, sliding_window_length)

for session_index in tqdm(range(len(collected_sessions)), desc="Training best models"):
    current_session_key = collected_sessions[session_index]["session_key"]
    training_sessions = {x["session_key"]: x for x in collected_sessions[:session_index+1]}
    training_session_keys = [x["session_key"] for x in collected_sessions[:session_index+1]]
    
    # sensor_trainer_pairs = [('pose_depth', 'svm'), ('pose_depth', 'random_forest'), ('pose_depth', 'knn'), ('pose_depth', 'xgboost'), ('pose_depth', 'balanced_rf'), ('pose_depth', 'balanced_bagging'), ('pose_depth', 'easy_ensemble'),
    #                         ('doppler', 'svm'), ('doppler', 'random_forest'), ('doppler', 'knn'), ('doppler', 'xgboost'), ('doppler', 'balanced_rf'), ('doppler', 'balanced_bagging'), ('doppler', 'easy_ensemble'),
    #                         ('thermal_vision', 'svm'), ('thermal_vision', 'random_forest'), ('thermal_vision', 'knn'), ('thermal_vision', 'xgboost'), ('thermal_vision', 'balanced_rf'), ('thermal_vision', 'balanced_bagging'), ('thermal_vision', 'easy_ensemble'),
    #                         ('thermal_audio', 'svm'), ('thermal_audio', 'random_forest'), ('thermal_audio', 'knn'), ('thermal_audio', 'xgboost'), ('thermal_audio', 'balanced_rf'), ('thermal_audio', 'balanced_bagging'), ('thermal_audio', 'easy_ensemble'),
    #                         ('persondepth_depth', 'svm'), ('persondepth_depth', 'random_forest'), ('persondepth_depth', 'knn'), ('persondepth_depth', 'xgboost'), ('persondepth_depth', 'balanced_rf'), ('persondepth_depth', 'balanced_bagging'), ('persondepth_depth', 'easy_ensemble'),
    #                         ('watchmotion_dominant', 'svm'), ('watchmotion_dominant', 'random_forest'), ('watchmotion_dominant', 'knn'), ('watchmotion_dominant', 'xgboost'), ('watchmotion_dominant', 'balanced_rf'), ('watchmotion_dominant', 'balanced_bagging'), ('watchmotion_dominant', 'easy_ensemble'),
    #                         ]

    sensor_trainer_pairs = [('pose_depth', 'svm'), ('pose_depth', 'knn'), ('pose_depth', 'xgboost'), ('pose_depth', 'balanced_rf'),
                            ('doppler', 'svm'), ('doppler', 'knn'), ('doppler', 'xgboost'), ('doppler', 'balanced_rf'),
                            ('thermal_vision', 'svm'), ('thermal_vision', 'knn'), ('thermal_vision', 'xgboost'), ('thermal_vision', 'balanced_rf'),
                            ('thermal_audio', 'svm'), ('thermal_audio', 'knn'), ('thermal_audio', 'xgboost'), ('thermal_audio', 'balanced_rf'),
                            ('persondepth_depth', 'svm'), ('persondepth_depth', 'knn'), ('persondepth_depth', 'xgboost'), ('persondepth_depth', 'balanced_rf'),
                            ('watchmotion_dominant', 'svm'), ('watchmotion_dominant', 'knn'), ('watchmotion_dominant', 'xgboost'), ('watchmotion_dominant', 'balanced_rf'),
                            ]
    if not sensor_trainer_pairs:
        print(f"No valid sensor-trainer pairs found in selection results for {current_session_key}")
        continue
    
    print(f"Session {current_session_key}: Training {len(sensor_trainer_pairs)} selected models")
    print(f"Selected sensor-trainer pairs: {sensor_trainer_pairs}")
    
    # Create output directory for this session's best models
    session_best_models_dir = f"{best_models_base_dir}/{current_session_key}"
    os.makedirs(session_best_models_dir, exist_ok=True)
    
    # Get training labels
    label_generation_dir = f"{base_results_dir}/{participant}/label_generation_{window_size}_{sliding_window_length}/{args.description_vlm_model}/{current_session_key}"
    location_action_file = os.path.join(label_generation_dir, f"location_action_labels_{args.clustering_vlm_model}.csv")
    if not os.path.exists(location_action_file):
        print(f"Location action file {location_action_file} does not exist. Skipping...")
        continue
    df_location_action_labels = pd.read_csv(location_action_file)
    df_location_action_labels = df_location_action_labels[['window_id', 'location_cluster','activity_cluster','confidence']]
    df_location_action_labels = df_location_action_labels[df_location_action_labels['location_cluster'] != 'no_match']
    df_location_action_labels = df_location_action_labels[df_location_action_labels['activity_cluster'] != 'no_match']
    df_location_action_labels = df_location_action_labels[df_location_action_labels['activity_cluster'].notna()]
    df_location_action_labels = df_location_action_labels[df_location_action_labels['location_cluster'].notna()]
    df_location_action_labels['location'] = df_location_action_labels['location_cluster'].astype(str)
    df_location_action_labels['activity'] = df_location_action_labels['activity_cluster'].astype(str)

    # get activity clusters file
    activity_clusters_file = os.path.join(label_generation_dir, f"optimal_activity_clusters_{args.alpha}.csv")
    if not os.path.exists(activity_clusters_file):
        print(f"Activity clusters file {activity_clusters_file} does not exist. Skipping...")
        continue
    df_activity_clusters = pd.read_csv(activity_clusters_file)
    df_activity_clusters['cluster_id'] = df_activity_clusters['cluster_id'].astype(int)
    df_activity_clusters['merged_label'] = df_activity_clusters['merged_label'].astype(str)

    # merge the location action labels with the activity clusters
    df_location_action_labels = df_location_action_labels.merge(df_activity_clusters, on=['location','activity'], how='left')
    df_location_action_labels = df_location_action_labels[df_location_action_labels['merged_label'].notna()]
    df_location_action_labels['location'] = df_location_action_labels['location'].astype(str)
    df_location_action_labels['activity'] = df_location_action_labels['activity'].astype(str)
    df_location_action_labels['merged_label'] = df_location_action_labels['merged_label'].astype(str)
    df_location_action_labels['final_label'] = df_location_action_labels.apply(lambda row: f"{row['cluster_id']}-{row['merged_label']}", axis=1)
    all_labels = sorted(df_location_action_labels['final_label'].unique())
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}
    num_classes = len(all_labels)
    
    # Create soft label vectors
    window_label_dict = {}
    for window_id, group in df_location_action_labels.groupby('window_id'):
        label_vec = np.zeros(num_classes, dtype=float)
        for _, row in group.iterrows():
            label_idx = label_to_index[row['final_label']]
            label_vec[label_idx] = row['confidence']
        if label_vec.sum() > 0:
            label_vec = label_vec / label_vec.sum()
        window_label_dict[window_id] = label_vec
    
    # Extract timestamps and corresponding soft label vectors
    label_end_timestamps = np.array(list(window_label_dict.keys()), dtype=float)
    training_window_labels = np.array(list(window_label_dict.values()))
    
    # Create window_id -> label mapping for all training windows
    training_label_mapping = {}
    
    for i, end_ts in enumerate(label_end_timestamps):
        ts_range = np.arange(end_ts - window_size, end_ts, sliding_window_length)
        for ts in ts_range:
            training_label_mapping[ts] = training_window_labels[i]
    
    # Load feature data for training sessions
    training_feature_results = {}
    
    for session_key in tqdm(training_sessions, desc="Loading features", leave=False):
        for sensor_name in allowed_sensors:
            sensor_path = f"{features_base_dir}/{session_key}_{sensor_name}.pkl"
            if sensor_name =="watchmotion_dominant":
                sensor_path = f"{features_base_dir}/{session_key}_watch_motion.pkl"
            if not os.path.exists(sensor_path):
                continue
            with open(sensor_path, "rb") as f:
                try:    
                    sensor_data = pickle.load(f)
                except Exception as e:
                    print(f"Error loading sensor data for {sensor_name} in {session_key}: {e}")
                    continue
            if len(sensor_data['features']) == 0:
                continue
            
            # Process features
            none_mask = [x is not None for x in sensor_data['features']]
            features = [x for x in sensor_data['features'] if x is not None]
            window_ids = [x for x, y in zip(sensor_data['timestamps'], none_mask) if y]
            
            df_sensor_data = pd.DataFrame.from_records(features, index=window_ids)
            df_sensor_data = df_sensor_data.fillna(0.0).astype(float)
            df_sensor_data = df_sensor_data.replace([np.inf, -np.inf], 0.0)
            # Only keep windows that have corresponding labels
            df_sensor_data = df_sensor_data[df_sensor_data.index.isin(training_label_mapping.keys())]
            
            if sensor_name not in training_feature_results:
                training_feature_results[sensor_name] = df_sensor_data
            else:
                training_feature_results[sensor_name] = pd.concat([training_feature_results[sensor_name], df_sensor_data], axis=0)
    
    # Train best models (complete training without cross-validation)
    for sensor_name, trainer_name in tqdm(sensor_trainer_pairs, desc="Training models", leave=False):
        if sensor_name not in training_feature_results:
            print(f"Sensor {sensor_name} not found in training features. Skipping...")
            continue
        
        model_output_file = f"{session_best_models_dir}/{sensor_name}_{trainer_name}_model.pkl"
        if os.path.exists(model_output_file):
            continue
        
        # Get training data by properly aligning features and labels using window_ids
        sensor_df = training_feature_results[sensor_name]
        
        # Find common window_ids between features and labels
        common_window_ids = set(sensor_df.index) & set(training_label_mapping.keys())
        common_window_ids = sorted(list(common_window_ids))
        
        if len(common_window_ids) == 0:
            print(f"No common window_ids found between features and labels for sensor {sensor_name}. Skipping...")
            continue
        
        # Create aligned training data
        X_train = sensor_df.loc[common_window_ids].values
        y_train = np.array([training_label_mapping[window_id] for window_id in common_window_ids])
        
        # print(f"Training {sensor_name}_{trainer_name} with {len(X_train)} samples (X_train shape: {X_train.shape}, y_train shape: {y_train.shape})")
        
        # Check if we have variation in training labels
        unique_label_patterns, counts = np.unique(y_train, axis=0, return_counts=True)
        
        if len(unique_label_patterns) == 1:
            # All samples have same label pattern - save this pattern as the model
            model_data = {
                'type': 'constant_prediction',
                'prediction_pattern': unique_label_patterns[0],
                'all_labels': all_labels,
                'label_to_index': label_to_index,
                'num_classes': num_classes
            }
        else:
            # Train the classifier
            trainer_clf = get_trainer(trainer_name)()
            try:
                trainer_clf.train(X_train, y_train, class_names=all_labels)
            except Exception as e:
                print(f"Error training {trainer_name} for {sensor_name}: {e}")
                print(traceback.format_exc())
                continue
            
            model_data = {
                'type': 'trained_classifier',
                'classifier': trainer_clf,
                'all_labels': all_labels,
                'label_to_index': label_to_index,
                'num_classes': num_classes
            }
        
        # Save the trained model
        with open(model_output_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        # print(f"Saved model: {sensor_name}_{trainer_name}")
    
    # Save metadata about this session's best models
    metadata = {
        'session_key': current_session_key,
        'selected_sensor_trainer_pairs': sensor_trainer_pairs,
        'selection_method': f"{args.direction}",
        'num_label_windows': len(training_label_mapping),
        'num_classes': num_classes,
        'all_labels': all_labels,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_file = f"{session_best_models_dir}/model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Completed training for session {current_session_key}")

print("Best model training completed!")