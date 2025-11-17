# This files looks at task videos from video dir and reorganize the files to kitchen | user | task | instance directory
import os
from datetime import datetime
import numpy as np
import argparse
import json
import sys
import pickle
import dotenv
dotenv.load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
sys.path.append(PROJECT_ROOT)

from src.featurization.pose.PoseFeatures import PoseFeatures
from src.featurization.depth.PersonDepthFeatures import PersonDepthFeatures
from src.featurization.doppler.DopplerFeatures import DopplerFeatures
from src.featurization.thermal.SimpleThermalFeaturizer import SimpleThermalFeaturizer

base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")
base_results_dir = os.environ.get("BASE_RESULTS_DIR", "./results")
os.makedirs(base_results_dir, exist_ok=True)

parser = argparse.ArgumentParser(description="Process video sessions.")
parser.add_argument("--participant", default="eu2a2-data-collection", help="Participant name")
parser.add_argument("--session_prefix", default="P12", help="Session prefix")
parser.add_argument("--window_size", default=5.0, help="Window size in seconds")
parser.add_argument("--sliding_window_length", default=0.5, help="Sliding window length in seconds")
parser.add_argument("--debug", action="store_true", help="Debug mode")
args = parser.parse_args()

participant = args.participant
session_prefix = args.session_prefix
window_size = args.window_size
sliding_window_length = args.sliding_window_length
debug = args.debug
# create the features base directory
features_base_dir = f"{base_results_dir}/{participant}/features_{window_size}_{sliding_window_length}"
os.makedirs(features_base_dir, exist_ok=True)

# get the video and depth directories
base_video_dir = f"{base_processed_dir}/{participant}/processed_video_data"
base_depth_dir = f"{base_processed_dir}/{participant}/processed_depth_data"
base_thermal_vision_dir = f"{base_processed_dir}/{participant}/processed_thermal-vision_data"
base_thermal_audio_dir = f"{base_processed_dir}/{participant}/processed_thermal-audio_data"

# get all session information
watch_sessions_file = f"{base_processed_dir}/{participant}/watch_sessions_filtered.json"
if not os.path.exists(watch_sessions_file):
    print(f"File {watch_sessions_file} does not exist. Exiting...")
    exit(1)

# get the watch files
with open(watch_sessions_file, "r") as f:
    watch_sessions = json.load(f)

# list all the collected sessions
def video_file_exists(base_filename, video_dir, depth_dir):
    # check if the file exists in the directory
    if os.path.exists(os.path.join(video_dir, base_filename)):
        return True
    if os.path.exists(os.path.join(depth_dir, base_filename)):
        return True
    return False


collected_sessions = {
        f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}": {
            **session_data,
        }
        for session_idx, session_data in enumerate(watch_sessions)
        if video_file_exists(f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}.mp4",base_video_dir, base_depth_dir)
    }

# get some paths for models
pose_model_name = "yolo11m-pose"
pose_cache_dir = os.environ.get("POSE_CACHE_DIR", "./models/pose")
depth_model_path = os.environ.get("DEPTH_MODEL_PATH", "./models/monodepth/DepthAnythingV2SmallF16.mlpackage")

pose_extractor = PoseFeatures()
depth_extractor = PersonDepthFeatures()
thermal_extractor = SimpleThermalFeaturizer()
doppler_extractor = DopplerFeatures()


# loop over session to process and save the data
for session_key in collected_sessions:

    # get the session start and end
    print(f"----------------------------------------Processing session {session_key}----------------------------------------")
    session_start = collected_sessions[session_key]['start']
    session_end = collected_sessions[session_key]['end']
    session_start_timestamp = datetime.strptime(session_start, "%Y%m%d_%H%M%S").timestamp()
    session_end_timestamp = datetime.strptime(session_end, "%Y%m%d_%H%M%S").timestamp()

    # get the session files for video based sensors
    birdseye_video_file = f"{base_video_dir}/{session_key}.mp4"
    depth_video_file = f"{base_depth_dir}/{session_key}.mp4"

    # get the windows for the session
    windows = pose_extractor.create_fixed_windows(session_start_timestamp, session_end_timestamp, window_size, sliding_window_length)
    all_timestamps = [window[-1] for window in windows]
    print(f"Number of windows: {len(all_timestamps)}, first window: {all_timestamps[0]}, last window: {all_timestamps[-1]}, duration: {all_timestamps[-1] - all_timestamps[0]} seconds")
    
    # get the features for birdseye video
    print(f"Processing birdseye video: {birdseye_video_file}...")
    pose_timestamps_birdseye, pose_features_birdseye = pose_extractor.generate_timestamp_windows(birdseye_video_file, windows, pose_model_name, pose_cache_dir, debug=debug)
    depth_timestamps_birdseye, depth_features_birdseye = depth_extractor.generate_timestamp_windows_from_video(depth_video_file, windows, depth_model_path, "yolo11m-seg", seg_cache_dir, debug=debug)
    pose_birdseye = {ts: pose_features_birdseye[i] for i, ts in enumerate(pose_timestamps_birdseye)}
    depth_birdseye = {ts: depth_features_birdseye[i] for i, ts in enumerate(depth_timestamps_birdseye)}
    
    # get the pose and depth features for depth video
    print(f"Processing depth video: {depth_video_file}...")
    pose_timestamps_depth, pose_features_depth = pose_extractor.generate_timestamp_windows(depth_video_file, windows, pose_model_name, pose_cache_dir, debug=debug)
    depth_timestamps_depth, depth_features_depth = depth_extractor.generate_timestamp_windows_from_video(depth_video_file, windows, depth_model_path, "yolo11m-seg", seg_cache_dir, debug=debug)
    pose_depth = {ts: pose_features_depth[i] for i, ts in enumerate(pose_timestamps_depth)}
    depth_depth = {ts: depth_features_depth[i] for i, ts in enumerate(depth_timestamps_depth)}

    # get the session files for thermal based sensors
    thermal_vision_file = f"{base_thermal_vision_dir}/{session_key}.mp4"
    thermal_audio_file = f"{base_thermal_audio_dir}/{session_key}.mp4"

    # get the features for thermal vision and audio
    print(f"Processing thermal: {thermal_vision_file}...")
    thermal_timestamps_vision, thermal_features_vision = thermal_extractor.generate_timestamp_windows_from_video(thermal_vision_file, windows, debug=debug)
    print(f"Processing thermal: {thermal_audio_file}...")
    thermal_timestamps_audio, thermal_features_audio = thermal_extractor.generate_timestamp_windows_from_video(thermal_audio_file, windows, debug=debug)
    thermal_vision = {ts: thermal_features_vision[i] for i, ts in enumerate(thermal_timestamps_vision)}
    thermal_audio = {ts: thermal_features_audio[i] for i, ts in enumerate(thermal_timestamps_audio)}

    # get the features for doppler
    print("Processing doppler...")
    input_session_dir = f"{base_processed_dir}/{participant}/sessions/{session_key.split('_')[0]}"
    output_session_dir = f"{base_results_dir}/{participant}/sessions/{session_key}"
    doppler_timestamps, doppler_raw_data = doppler_extractor.process_b64_doppler_data(input_session_dir, output_session_dir)
    # sort the timestamps and raw data
    sorted_indices = np.argsort(doppler_timestamps)
    doppler_timestamps = np.array(doppler_timestamps)[sorted_indices]
    doppler_raw_data = np.array(doppler_raw_data)[sorted_indices]   

    # get the features for doppler
    doppler_timestamps, doppler_features = doppler_extractor.generate_timestamp_windows(doppler_timestamps, doppler_raw_data, windows)
    doppler = {ts: doppler_features[i] for i, ts in enumerate(doppler_timestamps)}

    # map the features to the timestamps

    session_features = {
        "timestamp": all_timestamps,
        "pose_birdseye": [pose_birdseye.get(ts, None) for ts in all_timestamps],
        "depth_birdseye": [depth_birdseye.get(ts, None) for ts in all_timestamps],
        "pose_depth": [pose_depth.get(ts, None) for ts in all_timestamps],
        "depth_depth": [depth_depth.get(ts, None) for ts in all_timestamps],
        "thermal_vision": [thermal_vision.get(ts, None) for ts in all_timestamps],
        "thermal_audio": [thermal_audio.get(ts, None) for ts in all_timestamps],
        "doppler": [doppler.get(ts, None) for ts in all_timestamps],
        "config": {
            "pose_model_name": pose_model_name,
            "pose_cache_dir": pose_cache_dir,
            "depth_model_path": depth_model_path,
            "window_size": window_size,
            "sliding_window_length": sliding_window_length,     
        },
        "session_info": {
            "session_key": session_key,
            "session_start": session_start,
            "session_end": session_end,
        }
    }    

    # save the features using pickle
    with open(f"{features_base_dir}/{session_key}.pkl", "wb") as f:
        pickle.dump(session_features, f)
    print(f"----------------------------------------Finished processing session {session_key}----------------------------------------")

    
    
    