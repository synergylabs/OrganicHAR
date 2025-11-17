import os
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import json
import sys
import shutil
import pickle
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
sys.path.append(PROJECT_ROOT)

from src.key_moment_detection.GMMAnomalyDetector import GMMAnomalyDetection, fetch_topk_windows

base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")
existing_results_dir = os.environ.get("EXISTING_RESULTS_DIR", "./results")
new_results_dir = "/Volumes/Research-Prasoon/OrganicHAR/inhome_evaluation"
os.makedirs(new_results_dir, exist_ok=True)

parser = argparse.ArgumentParser(description="Process video sessions.")
parser.add_argument("--participant", default="P3-data-collection", help="Participant name")
parser.add_argument("--session_prefix", default="P3", help="Session prefix")
parser.add_argument("--window_size", default=5.0, help="Window size in seconds")
parser.add_argument("--sliding_window_length", default=0.5, help="Sliding window length in seconds")
parser.add_argument("--debug", action="store_true", help="Debug mode")
args = parser.parse_args()

allowed_sensors = ["pose_depth", "persondepth_depth", "pose_birdseye", "persondepth_birdseye", "doppler", "thermal_vision", "thermal_audio", "watchmotion_dominant", "watchmotion_nondominant"]

participant = args.participant
session_prefix = args.session_prefix
window_size = args.window_size
sliding_window_length = args.sliding_window_length
debug = args.debug
# create the features base directory
features_base_dir = f"{existing_results_dir}/{participant}/features_{window_size}_{sliding_window_length}"
new_features_base_dir = f"{new_results_dir}/{participant}/features_{window_size}_{sliding_window_length}"
os.makedirs(new_features_base_dir, exist_ok=True)

def get_session_data(base_processed_dir, participant, session_prefix):
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

collected_sessions = get_session_data(base_processed_dir, participant, session_prefix)


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
        pose_birdseye = f"{features_base_dir}/{session_key}_pose_birdseye.pkl" if "pose_birdseye" in allowed_sensors else None,
        pose_depth = f"{features_base_dir}/{session_key}_pose_depth.pkl" if "pose_depth" in allowed_sensors else None,
        thermal_vision = f"{features_base_dir}/{session_key}_thermal_vision.pkl" if "thermal_vision" in allowed_sensors else None,
        thermal_audio = f"{features_base_dir}/{session_key}_thermal_audio.pkl" if "thermal_audio" in allowed_sensors else None,
        doppler = f"{features_base_dir}/{session_key}_doppler.pkl" if "doppler" in allowed_sensors else None,
        persondepth_birdseye = f"{features_base_dir}/{session_key}_persondepth_birdseye.pkl" if "persondepth_birdseye" in allowed_sensors else None,
        persondepth_depth = f"{features_base_dir}/{session_key}_persondepth_depth.pkl" if "persondepth_depth" in allowed_sensors else None,
    )
    new_sensor_paths = dict(
        pose_birdseye = f"{new_features_base_dir}/{session_key}_pose_birdseye.pkl" if "pose_birdseye" in allowed_sensors else None,
        pose_depth = f"{new_features_base_dir}/{session_key}_pose_depth.pkl" if "pose_depth" in allowed_sensors else None,
        thermal_vision = f"{new_features_base_dir}/{session_key}_thermal_vision.pkl" if "thermal_vision" in allowed_sensors else None,
        thermal_audio = f"{new_features_base_dir}/{session_key}_thermal_audio.pkl" if "thermal_audio" in allowed_sensors else None,
        doppler = f"{new_features_base_dir}/{session_key}_doppler.pkl" if "doppler" in allowed_sensors else None,
        persondepth_birdseye = f"{new_features_base_dir}/{session_key}_persondepth_birdseye.pkl" if "persondepth_birdseye" in allowed_sensors else None,
        persondepth_depth = f"{new_features_base_dir}/{session_key}_persondepth_depth.pkl" if "persondepth_depth" in allowed_sensors else None,
    )
    
    all_feature_results = {}
    for sensor_name, sensor_path in session_paths.items():
        if sensor_path is None:
            continue
        if os.path.exists(new_sensor_paths[sensor_name]):
            print(f"File {new_sensor_paths[sensor_name]} already exists. Skipping...")
            continue
        if os.path.exists(session_paths[sensor_name]):
            shutil.copy(session_paths[sensor_name], new_sensor_paths[sensor_name])
            print(f"Copied {session_key} | {sensor_name} | {session_paths[sensor_name]} to {new_sensor_paths[sensor_name]}")
            continue
