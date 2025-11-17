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
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")
base_results_dir = "/Volumes/Research-Prasoon/OrganicHAR/inhome_evaluation"
os.makedirs(base_results_dir, exist_ok=True)

parser = argparse.ArgumentParser(description="Run predictions using best models from sequential selection.")
parser.add_argument("--participant", default="P5data-collection", help="Participant name")
parser.add_argument("--session_prefix", default="P11", help="Session prefix")
parser.add_argument("--window_size", default=5.0, help="Window size in seconds")
parser.add_argument("--sliding_window_length", default=0.5, help="Sliding window length in seconds")
parser.add_argument("--description_vlm_model", required=True, help="VLM model used for description generation")
parser.add_argument("--clustering_vlm_model", default="gpt-4.1", help="VLM model used for location and activity clustering")
parser.add_argument("--merging_threshold", default=0.3, help="Threshold for merging similar labels")
parser.add_argument("--label_humanizer_model", default="gpt-4.1-mini", help="VLM model used for label humanization")
parser.add_argument("--alpha", required=True, help="Alpha for activity clustering")
parser.add_argument("--ensemble_method", default="majority_vote", choices=["majority_vote", "average_confidence"], help="Ensemble method")
parser.add_argument("--direction", default="forward", choices=["forward", "backward"], help="Selection direction")
parser.add_argument("--debug", action="store_true", help="Debug mode")
args = parser.parse_args()

allowed_sensors = ["doppler", "pose_depth", "thermal_vision", "thermal_audio"]
allowed_sensors += ["persondepth_depth", "watchmotion_dominant"]

participant = args.participant
session_prefix = args.session_prefix
window_size = args.window_size
sliding_window_length = args.sliding_window_length
debug = args.debug

features_base_dir = f"{base_results_dir}/{participant}/features_{window_size}_{sliding_window_length}"
best_models_base_dir = f"{base_results_dir}/{participant}/best_models_v3_alpha_{args.alpha}_{window_size}_{sliding_window_length}/{args.description_vlm_model}/{args.merging_threshold}_control_{args.clustering_vlm_model}"
activity_predictions_base_dir = f"{base_results_dir}/{participant}/activity_predictions_v3_alpha_{args.alpha}_{window_size}_{sliding_window_length}/{args.description_vlm_model}/{args.merging_threshold}_control_{args.clustering_vlm_model}/{args.direction}"
os.makedirs(activity_predictions_base_dir, exist_ok=True)

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

def load_ground_truth_labels(base_results_dir, participant, window_size, sliding_window_length, description_vlm_model, current_session_key, merging_threshold, clustering_vlm_model):
    """Load ground truth labels if they exist for the session, return dict of window_id -> ground_truth_label"""
    label_generation_dir = f"{base_results_dir}/{participant}/label_generation_{window_size}_{sliding_window_length}/{description_vlm_model}/{current_session_key}/{merging_threshold}_control_{clustering_vlm_model}"
    training_labels_file = os.path.join(label_generation_dir, f"training_labels_{args.label_humanizer_model}.csv")
    
    if not os.path.exists(training_labels_file):
        return {}
    
    df_training_labels = pd.read_csv(training_labels_file)
    df_training_labels = df_training_labels[['window_id', 'training_label','confidence']]
    df_training_labels['training_label'] = df_training_labels['training_label'].astype(str)
    
    # For each window, get the label with highest confidence
    ground_truth_dict = {}
    for window_id, group in df_training_labels.groupby('window_id'):
        max_conf_row = group.loc[group['confidence'].idxmax()]
        ground_truth_dict[window_id] = max_conf_row['training_label']
    
    return ground_truth_dict

def predict_with_model(model_data, X):
    """Make predictions using loaded model"""
    if model_data['type'] == 'constant_prediction':
        # Return the constant prediction pattern for all samples
        prediction_pattern = model_data['prediction_pattern']
        return np.tile(prediction_pattern, (len(X), 1))
    
    elif model_data['type'] == 'trained_classifier':
        classifier = model_data['classifier']
        
        # Try predict_proba first
        try:
            if hasattr(classifier, 'predict_proba'):
                prob_results = classifier.predict_proba(X)
                num_classes = model_data['num_classes']
                all_labels = model_data['all_labels']
                
                predicted_probabilities = np.zeros((len(X), num_classes))
                
                # Handle different return formats
                if isinstance(prob_results, dict):
                    prob_results = [prob_results]
                elif isinstance(prob_results, list) and len(prob_results) > 0 and isinstance(prob_results[0], dict):
                    pass
                else:
                    raise ValueError(f"Unexpected format from predict_proba: {type(prob_results)}")
                
                # Convert to numpy array
                for i, prob_dict in enumerate(prob_results):
                    for j, label in enumerate(all_labels):
                        if label in prob_dict:
                            predicted_probabilities[i, j] = prob_dict[label]
                        else:
                            predicted_probabilities[i, j] = 0.0
                
                return predicted_probabilities
            else:
                # Fall back to regular predictions
                predictions, confidences = classifier.predict(X)
                predicted_probabilities = np.zeros((len(predictions), model_data['num_classes']))
                for i, pred in enumerate(predictions):
                    if pred in model_data['label_to_index']:
                        predicted_probabilities[i, model_data['label_to_index'][pred]] = confidences[i]
                return predicted_probabilities
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Fall back to regular predictions
            predictions, confidences = classifier.predict(X)
            predicted_probabilities = np.zeros((len(predictions), model_data['num_classes']))
            for i, pred in enumerate(predictions):
                if pred in model_data['label_to_index']:
                    predicted_probabilities[i, model_data['label_to_index'][pred]] = confidences[i]
            return predicted_probabilities

collected_sessions = get_session_data(base_processed_dir, participant, session_prefix, window_size, sliding_window_length)


# Load all available feature data
all_feature_results = {}
all_session_keys = [x["session_key"] for x in collected_sessions]

for session_key in tqdm(all_session_keys, desc="Loading all features", leave=False):
    for sensor_name in allowed_sensors:
        sensor_path = f"{features_base_dir}/{session_key}_{sensor_name}.pkl"
        if not os.path.exists(sensor_path):
            continue
        with open(sensor_path, "rb") as f:
            try:
                sensor_data = pickle.load(f)
            except Exception as e:
                print(f"Error loading sensor data for {sensor_path}: {e}")
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
        
        session_sensor_key = f"{session_key}_{sensor_name}"
        all_feature_results[session_sensor_key] = df_sensor_data
        
for session_index in tqdm(range(len(collected_sessions)), desc="Running predictions"):
    current_session_key = collected_sessions[session_index]["session_key"]
    
    # Check if best models exist for this session
    session_best_models_dir = f"{best_models_base_dir}/{current_session_key}"
    if not os.path.exists(session_best_models_dir):
        print(f"Best models directory {session_best_models_dir} does not exist. Skipping...")
        continue
    
    # Load model metadata
    metadata_file = f"{session_best_models_dir}/model_metadata.json"
    if not os.path.exists(metadata_file):
        print(f"Model metadata file {metadata_file} does not exist. Skipping...")
        continue
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    sensor_trainer_pairs = metadata['selected_sensor_trainer_pairs']
    all_labels = metadata['all_labels']
    num_classes = metadata['num_classes']
    
    print(f"Session {current_session_key}: Running predictions for {len(sensor_trainer_pairs)} models")
    
    # Create output directory for this session's predictions
    session_predictions_dir = f"{activity_predictions_base_dir}/{current_session_key}"
    os.makedirs(session_predictions_dir, exist_ok=True)
    
    # Load ground truth labels if available
    ground_truth_dict = load_ground_truth_labels(
        base_results_dir, participant, window_size, sliding_window_length, 
        args.description_vlm_model, current_session_key, args.merging_threshold, args.clustering_vlm_model
    )
    

    # Run predictions for each model
    for sensor_name, trainer_name in tqdm(sensor_trainer_pairs, desc="Running model predictions", leave=False):
        model_file = f"{session_best_models_dir}/{sensor_name}_{trainer_name}_model.pkl"
        if not os.path.exists(model_file):
            print(f"Model file {model_file} does not exist. Skipping...")
            continue
        
        try:
            # Load the trained model
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading model file {model_file}: {e}")
            continue
        
        # Prepare unified results list
        all_predictions = []

        # Check if predictions file already exists
        predictions_file = f"{session_predictions_dir}/{sensor_name}_{trainer_name}-predictions.csv"
        if os.path.exists(predictions_file):
            print(f"Predictions for {sensor_name}_{trainer_name} already exist. Skipping...")
            continue
        
        # Generate predictions for all sessions that have features for this sensor
        for session_sensor_key, feature_data in all_feature_results.items():
            if not session_sensor_key.endswith(f"_{sensor_name}"):
                continue
                
            # Extract session key from the feature key
            feature_session_key = session_sensor_key.replace(f"_{sensor_name}", "")
            
            X = feature_data.values
            window_ids = feature_data.index.values
            
            if len(X) == 0:
                continue
            
            # Make predictions
            predicted_probabilities = predict_with_model(model_data, X)
            
            # Save results
            for i, window_id in enumerate(window_ids):
                pred_probs = predicted_probabilities[i]
                max_prob_idx = np.argmax(pred_probs)
                main_prediction = all_labels[max_prob_idx]
                main_confidence = pred_probs[max_prob_idx]
                
                result_entry = {
                    "session_key": feature_session_key,
                    "sensor_name": sensor_name,
                    "trainer": trainer_name,
                    "window_id": window_id,
                    "main_prediction": main_prediction,
                    "main_confidence": main_confidence,
                }
                
                # Add prediction probabilities
                for j, label in enumerate(all_labels):
                    result_entry[f"pred_{label}"] = pred_probs[j]
                
                # Add ground truth if available (only for current session being processed)
                if feature_session_key == current_session_key:
                    result_entry["gt_label"] = ground_truth_dict.get(window_id, "")
                    result_entry["gt_confidence"] = 1.0 if result_entry["gt_label"] else 0.0
                else:
                    result_entry["gt_label"] = ""
                    result_entry["gt_confidence"] = 0.0
                
                all_predictions.append(result_entry)
        
        # Save unified predictions
        if all_predictions:
            pd.DataFrame(all_predictions).to_csv(predictions_file, index=False)
        
        # print(f"Saved predictions for {sensor_name}_{trainer_name}: {len(all_predictions)} total")
    
    # Save session metadata
    session_metadata = {
        'session_key': current_session_key,
        'sensor_trainer_pairs': sensor_trainer_pairs,
        'num_classes': num_classes,
        'all_labels': all_labels,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_file = f"{session_predictions_dir}/prediction_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(session_metadata, f, indent=2)
    
    # Generate ensemble predictions using the selected sensor-trainer pairs
    print(f"Generating ensemble predictions for session {current_session_key}")
    print(f"Using {len(sensor_trainer_pairs)} selected models: {[f'{s}_{t}' for s, t in sensor_trainer_pairs]}")
    
    # Load all individual predictions for selected models
    all_predictions = {}
    
    for sensor_name, trainer_name in sensor_trainer_pairs:
        predictions_file = f"{session_predictions_dir}/{sensor_name}_{trainer_name}-predictions.csv"
        
        if os.path.exists(predictions_file):
            df_pred = pd.read_csv(predictions_file)
            all_predictions[(sensor_name, trainer_name)] = df_pred
    
    # Process ensemble predictions (unified approach)
    ensemble_results = []
    if all_predictions:
        # Combine all predictions into a single DataFrame for vectorized operations
        combined_dfs = []
        for (sensor_name, trainer_name), df in all_predictions.items():
            df_copy = df.copy()
            df_copy['model_id'] = f"{sensor_name}_{trainer_name}"
            combined_dfs.append(df_copy)
        
        if combined_dfs:
            combined_df = pd.concat(combined_dfs, ignore_index=True)
            
            # Get probability columns (now using pred_ prefix)
            prob_cols = [f'pred_{label}' for label in all_labels]
            
            # Group by session_key and window_id to aggregate
            grouped = combined_df.groupby(['session_key', 'window_id'])
            
            for (session_key, window_id), group in tqdm(grouped, desc="Processing ensemble predictions", leave=False):
                # Extract probability matrix (models x classes)
                prob_matrix = group[prob_cols].values
                
                # # Sum probabilities across models and normalize
                # ensemble_probs = np.sum(prob_matrix, axis=0)
                # ensemble_probs = ensemble_probs / ensemble_probs.sum()
                
                # # Get prediction with highest confidence
                # max_idx = np.argmax(ensemble_probs)
                # prediction = all_labels[max_idx]
                # confidence = ensemble_probs[max_idx]

                # get the prediction based on majority vote
                pair_prediction_labels = prob_matrix.argmax(axis=1)
                pair_prediction_confidences = prob_matrix.max(axis=1)
                majority_prediction = np.bincount(pair_prediction_labels).argmax()
                majority_confidence = np.mean(pair_prediction_confidences[pair_prediction_labels == majority_prediction])
                prediction = all_labels[majority_prediction]
                confidence = majority_confidence

                gt_label = group.iloc[0]['gt_label']
                gt_confidence = group.iloc[0]['gt_confidence']
                
                ensemble_results.append({
                    'session_key': session_key,
                    'window_id': window_id,
                    'prediction': prediction,
                    'confidence': confidence,
                    'gt_label': gt_label,
                    'gt_confidence': gt_confidence
                })
    
    # Save ensemble predictions
    ensemble_predictions_dir = f"{session_predictions_dir}/ensemble_predictions"
    os.makedirs(ensemble_predictions_dir, exist_ok=True)
    
    if ensemble_results:
        ensemble_df = pd.DataFrame(ensemble_results)
        ensemble_file = f"{ensemble_predictions_dir}/predictions.csv"
        ensemble_df.to_csv(ensemble_file, index=False)
    
    print(f"Completed predictions for session {current_session_key}")

print("Prediction generation completed!")