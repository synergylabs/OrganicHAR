import os
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import json
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from common.anomaly_detection.GMMAnomalyDetector import GMMAnomalyDetection, fetch_topk_windows


base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")
base_results_dir = os.environ.get("BASE_RESULTS_DIR", "./results")
os.makedirs(base_results_dir, exist_ok=True)

parser = argparse.ArgumentParser(description="Process video sessions.")
parser.add_argument("--participant", default="RHP05", help="Participant name")
parser.add_argument("--session_prefix", default="P15", help="Session prefix")
parser.add_argument("--window_size", default=5.0, help="Window size in seconds")
parser.add_argument("--sliding_window_length", default=0.5, help="Sliding window length in seconds")
parser.add_argument("--num_detections_per_min", default=1, help="Number of anomalies to be detected per minute")
parser.add_argument("--debug", action="store_true", help="Debug mode")
args = parser.parse_args()

allowed_sensors = ["pose_depth","persondepth_depth","dinov2_depth","pose_birdseye","persondepth_birdseye","dinov2_birdseye", "watchmotion_dominant", "watchaudio_dominant"]
# allowed_sensors = ['watchaudio_dominant']
participant = args.participant
session_prefix = args.session_prefix
window_size = args.window_size
sliding_window_length = args.sliding_window_length
debug = args.debug
num_detection_per_min = args.num_detections_per_min
# create the features base directory
features_base_dir = f"{base_results_dir}/{participant}/features_{window_size}_{sliding_window_length}"
change_detection_base_dir = f"{base_results_dir}/{participant}/change_detection_{window_size}_{sliding_window_length}"
os.makedirs(change_detection_base_dir, exist_ok=True)

def get_session_data(base_processed_dir, participant, session_prefix, window_size, sliding_window_length):
    # get all session information
    watch_sessions_file = f"{base_processed_dir}/{participant}/watch_sessions_filtered.json"
    if not os.path.exists(watch_sessions_file):
        print(f"File {watch_sessions_file} does not exist. Exiting...")
        exit(1)
    with open(watch_sessions_file, "r") as f:
        watch_sessions = json.load(f)
    # get the video and depth directories
    base_video_dir = f"{base_processed_dir}/{participant}/processed_video_data"
    base_depth_dir = f"{base_processed_dir}/{participant}/processed_depth_data"
    base_thermal_vision_dir = f"{base_processed_dir}/{participant}/processed_thermal-vision_data"
    base_thermal_audio_dir = f"{base_processed_dir}/{participant}/processed_thermal-audio_data"

    def video_file_exists(base_filename, video_dir, depth_dir):
        # check if the file exists in the directory
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

collected_sessions = get_session_data(base_processed_dir, participant, session_prefix, window_size, sliding_window_length)


# loop over session to cluster and save the data
for session_index in range(len(collected_sessions)):
    session_key = collected_sessions[session_index]["session_key"]
    session_data = collected_sessions[session_index]
    session_start = session_data['start']
    session_end = session_data['end']
    session_start_timestamp = datetime.strptime(session_start, "%Y%m%d_%H%M%S").timestamp()
    session_end_timestamp = datetime.strptime(session_end, "%Y%m%d_%H%M%S").timestamp()

    # get the feature results
    session_paths = dict(
        pose_birdseye = f"{features_base_dir}/{session_key}_pose_birdseye.pkl" if 'pose_birdseye' in allowed_sensors else None,
        pose_depth = f"{features_base_dir}/{session_key}_pose_depth.pkl" if 'pose_depth' in allowed_sensors else None,
        thermal_vision = f"{features_base_dir}/{session_key}_thermal_vision.pkl" if 'thermal_vision' in allowed_sensors else None,
        thermal_audio = f"{features_base_dir}/{session_key}_thermal_audio.pkl" if 'thermal_audio' in allowed_sensors else None,
        doppler = f"{features_base_dir}/{session_key}_doppler.pkl" if 'doppler' in allowed_sensors else None,
        micarray = f"{features_base_dir}/{session_key}_micarray.pkl" if 'micarray' in allowed_sensors else None,
        persondepth_birdseye = f"{features_base_dir}/{session_key}_persondepth_birdseye.pkl" if 'persondepth_birdseye' in allowed_sensors else None,
        persondepth_depth = f"{features_base_dir}/{session_key}_persondepth_depth.pkl" if 'persondepth_depth' in allowed_sensors else None,
        dinov2_birdseye = f"{features_base_dir}/{session_key}_dinov2_birdseye.pkl" if 'dinov2_birdseye' in allowed_sensors else None,
        dinov2_depth = f"{features_base_dir}/{session_key}_dinov2_depth.pkl" if 'dinov2_depth' in allowed_sensors else None,
        watchmotion_dominant = f"{features_base_dir}/{session_key}_watch_motion.pkl" if 'watchmotion_dominant' in allowed_sensors else None,
        watchmotion_nondominant = f"{features_base_dir}/{session_key}_watch_motion.pkl" if 'watchmotion_nondominant' in allowed_sensors else None,
        watchaudio_dominant = f"{features_base_dir}/{session_key}_watch_audio.pkl" if 'watchaudio_dominant' in allowed_sensors else None,
        watchaudio_nondominant = f"{features_base_dir}/{session_key}_watch_audio.pkl" if 'watchaudio_nondominant' in allowed_sensors else None,
    )
    all_feature_results = {}
    for sensor_name, sensor_path in session_paths.items():
        if sensor_path is None:
            continue
        if not os.path.exists(sensor_path):
            print(f"File {sensor_path} does not exist. Exiting...")
            continue
        with open(sensor_path, "rb") as f:
            sensor_data = pickle.load(f)
        if len(sensor_data['features']) == 0:
            print(f"No features found for {sensor_name} in {session_key}. Continuing...")
            continue
        # remove any None feature values
        none_mask = [x is not None for x in sensor_data['features']]
        if sensor_name=="watchaudio_dominant":
            sensor_data['features'] = [np.array(x).flatten() for x in sensor_data['features'] if x is not None]
        else:
            sensor_data['features'] = [x for x in sensor_data['features'] if x is not None]
        sensor_data['timestamps'] = [x for x, y in zip(sensor_data['timestamps'], none_mask) if y]
        df_sensor_data = pd.DataFrame.from_records(sensor_data['features'], index=sensor_data['timestamps'])
        all_feature_results[sensor_name] = df_sensor_data.sort_index()



    for sensor_name in all_feature_results.keys():
        print(f"Detecting Anomalies {sensor_name}...")
        change_results_path = f"{change_detection_base_dir}/{session_key}_{sensor_name}_top_{num_detection_per_min}.csv"
        if os.path.exists(change_results_path):
            continue

        sensor_features = all_feature_results[sensor_name].values
        sensor_ts = all_feature_results[sensor_name].index

        sensor_ts_filled = np.arange(sensor_ts[0], sensor_ts[-1]+1, sliding_window_length)
        if sensor_ts_filled.shape[0]<=0:
            continue
        df_interpolated = pd.DataFrame(index=sensor_ts_filled)
        df_interpolated = pd.merge(df_interpolated, all_feature_results[sensor_name], left_index=True, right_index=True, how='left')
        df_interpolated = df_interpolated.interpolate(method='linear')
        sensor_features_filled = df_interpolated.values
        anomaly_detector = GMMAnomalyDetection(window_sec=sliding_window_length * 10,
                skip_sec=10)
        events = anomaly_detector(
                    features=sensor_features_filled,
                    dimensions=[{'name': sensor_name, 'first': 0, 'last': sensor_features_filled.shape[1]}],
                    fps=1/sliding_window_length,
                )
        scores = list(map(lambda x: x['score'], events))
        timestamps = list(map(lambda x: x['start'], events))
        
        # Data-driven top-k selection using statistical outlier detection
        scores_array = np.array(scores)
        
        # Method 1: IQR-based outlier detection (less strict)
        q1 = np.percentile(scores_array, 25)
        q3 = np.percentile(scores_array, 75)
        iqr = q3 - q1
        outlier_threshold = q1 - 1.0 * iqr  # Reduced from 1.5 to 1.0
        
        # Method 2: Standard deviation based (less strict)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        std_threshold = mean_score - 1.5 * std_score  # Reduced from 2 to 1.5
        
        # Method 3: Percentile-based (bottom 20% instead of 10%)
        percentile_threshold = np.percentile(scores_array, 20)
        
        # Use a less restrictive approach - take the median of the three thresholds
        # instead of the minimum (most restrictive)
        thresholds = [outlier_threshold, std_threshold, percentile_threshold]
        final_threshold = np.median(thresholds)
        
        # Count how many scores are below this threshold
        outlier_count = np.sum(scores_array < final_threshold)
        
        # Ensure we have at least 1 and at most 20 anomalies (reasonable bounds)
        top_k_val = max(1, min(outlier_count, 20))
        
        # print(f"Score statistics for {sensor_name}:")
        # print(f"  Mean: {mean_score:.4f}, Std: {std_score:.4f}")
        # print(f"  IQR threshold: {outlier_threshold:.4f}")
        # print(f"  Std threshold: {std_threshold:.4f}")
        # print(f"  Percentile threshold: {percentile_threshold:.4f}")
        # print(f"  Final threshold: {final_threshold:.4f}")
        print(f"  Selected top-k: {top_k_val}")
        
        topk_results = fetch_topk_windows(events, top_k_val)
        topk_frames = np.array(list(map(lambda x: x['min'][0]['frame'], topk_results))).astype(int)
        topk_scores = np.array(list(map(lambda x: x['score'], topk_results)))
        topk_timestamps = sensor_ts_filled[topk_frames]
        
        # Create enhanced plot with actual sensor timestamps
        plt.figure(figsize=(15, 6))
        
        # Convert sensor timestamps to datetime for better x-axis formatting
        sensor_timestamps_dt = [datetime.fromtimestamp(ts) for ts in sensor_ts_filled]
        topk_timestamps_dt = [datetime.fromtimestamp(ts) for ts in topk_timestamps]
        
        # Create scores array aligned with sensor timestamps (need to map GMM window scores to sensor frames)
        scores_aligned = np.full(len(sensor_ts_filled), np.nan)
        for i, event in enumerate(events):
            start_frame = int(event['start'] * (1/sliding_window_length))
            end_frame = min(start_frame + int(window_size * (1/sliding_window_length)), len(sensor_ts_filled))
            scores_aligned[start_frame:end_frame] = scores[i]
        
        # Plot the anomaly scores over time
        valid_mask = ~np.isnan(scores_aligned)
        plt.plot(np.array(sensor_timestamps_dt)[valid_mask], scores_aligned[valid_mask], 
                'b-', linewidth=1, alpha=0.7, label='Anomaly Scores')
        
        # Mark the selected anomaly times
        for i, (ts_dt, score) in enumerate(zip(topk_timestamps_dt, topk_scores)):
            plt.axvline(x=ts_dt, color='red', linestyle='--', alpha=0.8, linewidth=2)
            plt.scatter(ts_dt, score, color='red', s=100, zorder=5, 
                       label='Selected Anomalies' if i == 0 else "")
        
        # Formatting
        plt.xlabel('Time')
        plt.ylabel('Anomaly Score (lower = more anomalous)')
        plt.title(f'Anomaly Detection Results: {session_key} - {sensor_name}\n'
                 f'Top {len(topk_results)} anomalies selected')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the enhanced plot
        plt.savefig(f"{change_detection_base_dir}/{session_key}_{sensor_name}_scores.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        df_anomaly_results = pd.DataFrame(index=topk_timestamps, data={'score': topk_scores})
        df_anomaly_results.to_csv(change_results_path)