# This files looks at task videos from video dir and reorganize the files to kitchen | user | task | instance directory
import os
from datetime import datetime
import numpy as np
import argparse
import json
import sys
import pickle
import concurrent.futures
import threading
from functools import partial
import time
import glob
import tqdm
import pytz

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from sensors.watchmotion.core.featurizers.MotionFeaturizer import MotionFeaturizer
from sensors.watchaudio.core.featurizers.ASTFeaturizer import ASTFeaturizer



base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")
base_results_dir = os.environ.get("BASE_RESULTS_DIR", "./results")

os.makedirs(base_results_dir, exist_ok=True)

parser = argparse.ArgumentParser(description="Process video sessions.")
parser.add_argument("--participant", default="RHP05", help="Participant name")
parser.add_argument("--session_prefix", default="P15", help="Session prefix")
parser.add_argument("--window_size", default=2.5, help="Window size in seconds")
parser.add_argument("--sliding_window_length", default=0.25, help="Sliding window length in seconds")
parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument("--watch_type", default="android", help="Watch type (android or iwatch)")
parser.add_argument("--max_workers", default=4, type=int, help="Maximum number of parallel workers")
parser.add_argument("--tz_rpi", default="Europe/Lisbon", help="Timezone of the RPi")
parser.add_argument("--tz_watch", default="Europe/Lisbon", help="Timezone of the watch")

args = parser.parse_args()

participant = args.participant
session_prefix = args.session_prefix
window_size = float(args.window_size)
sliding_window_length = float(args.sliding_window_length)
debug = args.debug
max_workers = args.max_workers
watch_type = args.watch_type

allowed_sensors = ["watchmotion_dominant","watchmotion_nondominant"]
allowed_sensors += ["watchaudio_dominant","watchaudio_nondominant"]

# create the features base directory
features_base_dir = f"{base_results_dir}/{participant}/features_{window_size}_{sliding_window_length}"
os.makedirs(features_base_dir, exist_ok=True)

# get the video dir to check if the session exists
base_video_dir = f"{base_processed_dir}/{participant}/processed_video_data"
base_depth_dir = f"{base_processed_dir}/{participant}/processed_depth_data"

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



# get the model names
ast_model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
ast_coreml_cache_dir = os.environ.get("AST_COREML_CACHE_DIR", "./models/audio")

# Thread-local storage for extractors to avoid potential thread safety issues
thread_local_data = threading.local()

def tz_diff(date_str, tz1, tz2):
    """Return seconds to add to tz2 to get tz1 time"""
    dt = datetime.strptime(date_str, '%Y%m%d')
    tz1_time = pytz.timezone(tz1).localize(dt)
    tz2_time = pytz.timezone(tz2).localize(dt)
    return int((tz2_time - tz1_time).total_seconds())

def get_extractors():
    """Get thread-local instances of extractors"""
    if not hasattr(thread_local_data, 'extractors'):
        thread_local_data.extractors = {
            'watch-motion': MotionFeaturizer(),
            'watch-audio': ASTFeaturizer(model_name=ast_model_name, cache_dir=ast_coreml_cache_dir)
        }
    return thread_local_data.extractors

def process_watch_motion(input_session_dir, hand_type, windows, debug, save_path, watch_type, delta_seconds):

    # Check if output already exists
    if os.path.exists(save_path):
        print(f"[Thread {threading.current_thread().name}] Skipping watch-motion - output already exists: {save_path}")
        return 'watch-motion'
    
    print(f"[Thread {threading.current_thread().name}] Processing watch-motion: {input_session_dir}")
    extractors = get_extractors()
    start_time = time.time()

    # find all the pair of accel, gyro, mag files prefix
    watch_prefixes = glob.glob(f"{input_session_dir}/watch-{hand_type}*")
    # get the timestamp for the watch prefix
    watch_prefix_timestamps = set()
    for watch_prefix in watch_prefixes:
        if watch_type == "android":
            if "watch_recording" in watch_prefix:
                watch_prefix_timestamp = "_".join(watch_prefix.split("/")[-1].split("_")[1:-1])
            else:
                watch_prefix_timestamp = int(watch_prefix.split("/")[-1].split("_")[1])
        elif watch_type == "iwatch":
            watch_prefix_name = watch_prefix.split("/")[-1].split("_")[1]
            watch_prefix_date = watch_prefix.split("/")[-1].split("_")[2]
            watch_prefix_time = watch_prefix.split("/")[-1].split("_")[3]
            watch_prefix_timestamp = f"{watch_prefix_name}_{watch_prefix_date}_{watch_prefix_time}"
        watch_prefix_timestamps.add(watch_prefix_timestamp)
    
    watch_prefix_timestamps = sorted(list(watch_prefix_timestamps))
    all_timestamps = []
    all_features = []
    for watch_prefix_timestamp in watch_prefix_timestamps:
        timestamps, features = extractors['watch-motion'].generate_timestamp_windows(
            f"{input_session_dir}/watch-{hand_type}_{watch_prefix_timestamp}", windows, debug=debug, watch_type=watch_type, delta_seconds=delta_seconds
        )
        all_timestamps.extend(timestamps)
        all_features.extend(features)
    
    if len(all_timestamps) == 0:
        print(f"[Thread {threading.current_thread().name}] No watch-motion data found for {input_session_dir}")
        return f'watchmotion_{hand_type}'
    else:
        # Save directly to file
        watch_motion_data = {
            "timestamps": all_timestamps,
            "features": all_features,
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(watch_motion_data, f)
        
        end_time = time.time()
        print(f"[Thread {threading.current_thread().name}] Finished watch-motion in {end_time - start_time:.2f}s, extracted {len(all_timestamps)} windows")
        return f'watchmotion_{hand_type}'


def process_watch_audio(input_session_dir, hand_type, windows, debug, save_path, delta_seconds):
    
    # Check if output already exists
    if os.path.exists(save_path):
        print(f"[Thread {threading.current_thread().name}] Skipping watch-audio - output already exists: {save_path}")
        return 'watch-audio'
    
    print(f"[Thread {threading.current_thread().name}] Processing watch-audio: {input_session_dir}")
    extractors = get_extractors()
    start_time = time.time()
    
    # find all the pair of accel, gyro, mag files prefix
    watch_prefixes = glob.glob(f"{input_session_dir}/watch-{hand_type}*")
    # get the timestamp for the watch prefix
    watch_prefix_timestamps = set()
    for watch_prefix in watch_prefixes:
        if watch_type == "android":
            if "watch_recording" in watch_prefix:
                watch_prefix_timestamp = "_".join(watch_prefix.split("/")[-1].split("_")[1:-1])
            else:
                watch_prefix_timestamp = int(watch_prefix.split("/")[-1].split("_")[1])
        elif watch_type == "iwatch":
            watch_prefix_date = watch_prefix.split("/")[-1].split("_")[2]
            watch_prefix_time = watch_prefix.split("/")[-1].split("_")[3]
            watch_prefix_timestamp = datetime.strptime(f"{watch_prefix_date}_{watch_prefix_time}", "%y-%m-%d_%H-%M-%S").timestamp()
        watch_prefix_timestamps.add(watch_prefix_timestamp)
    
    watch_prefix_timestamps = sorted(list(watch_prefix_timestamps))
    all_timestamps = []
    all_features = []
    all_logit_features = []
    for watch_prefix_timestamp in watch_prefix_timestamps:
        timestamps, features, logit_features = extractors['watch-audio'].generate_timestamp_windows(
            f"{input_session_dir}/watch-{hand_type}_{watch_prefix_timestamp}", windows, debug=debug, delta_seconds=delta_seconds
        )
        all_timestamps.extend(timestamps)
        all_features.extend(features)
        all_logit_features.extend(logit_features)
    
    if len(all_timestamps) == 0:
        print(f"[Thread {threading.current_thread().name}] No watch-audio data found for {input_session_dir}")
        return f'watchaudio_{hand_type}'
    else:
        # Save directly to file
        watch_audio_data = {
            "timestamps": all_timestamps,
            "features": all_features,
            "logit_features": all_logit_features,
        }
        with open(save_path, "wb") as f:
            pickle.dump(watch_audio_data, f)
        
        end_time = time.time()
        print(f"[Thread {threading.current_thread().name}] Finished watch-audio in {end_time - start_time:.2f}s, extracted {len(all_timestamps)} windows")
        return f'watchaudio_{hand_type}'



# loop over session to process and save the data
for session_key in collected_sessions:
    print(f"----------------------------------------Processing session {session_key}----------------------------------------")
    session_start_time = time.time()
    
    # get the session start and end
    session_start = collected_sessions[session_key]['start']
    session_end = collected_sessions[session_key]['end']
    session_start_timestamp = datetime.strptime(session_start, "%Y%m%d_%H%M%S").timestamp()
    session_end_timestamp = datetime.strptime(session_end, "%Y%m%d_%H%M%S").timestamp()

    # get the windows for the session
    watch_motion_extractor = MotionFeaturizer()  # Create a temporary instance for window creation
    windows = watch_motion_extractor.create_fixed_windows(session_start_timestamp, session_end_timestamp, window_size, sliding_window_length)
    # get the delta between the windows and the supposedly watch time
    delta_seconds = tz_diff(session_start.split("_")[0], args.tz_rpi, args.tz_watch)
    
    all_timestamps = [window[-1] for window in windows]
    print(f"Number of windows: {len(all_timestamps)}, first window: {all_timestamps[0]}, last window: {all_timestamps[-1]}, duration: {all_timestamps[-1] - all_timestamps[0]} seconds")
    
    # Prepare tasks for parallel processing with save paths
    tasks = []
    
    # Create save paths for each task
    watch_motion_path = f"{features_base_dir}/{session_key}_watch_motion.pkl"
    watch_audio_path = f"{features_base_dir}/{session_key}_watch_audio.pkl"
    
    input_session_dir = f"{base_processed_dir}/{participant}/sessions/{'-'.join(session_key.split('-')[:2])}"
    output_session_dir = f"{base_results_dir}/{participant}/sessions/{session_key}"
    # print(f"Input session dir: {input_session_dir}, Output session dir: {output_session_dir}")
    os.makedirs(output_session_dir, exist_ok=True)
    watch_dominant_motion_task = partial(process_watch_motion, input_session_dir, "dominant", windows, debug, watch_motion_path, watch_type, delta_seconds)
    watch_nondominant_motion_task = partial(process_watch_motion, input_session_dir, "nondominant", windows, debug, watch_motion_path, watch_type, delta_seconds)
    watch_dominant_audio_task = partial(process_watch_audio, input_session_dir, "dominant", windows, debug, watch_audio_path, delta_seconds)
    watch_nondominant_audio_task = partial(process_watch_audio, input_session_dir, "nondominant", windows, debug, watch_audio_path, delta_seconds)

    tasks = [
        watch_dominant_motion_task if "watchmotion_dominant" in allowed_sensors else None,
        watch_nondominant_motion_task if "watchmotion_nondominant" in allowed_sensors else None,
        watch_dominant_audio_task if "watchaudio_dominant" in allowed_sensors else None,
        watch_nondominant_audio_task if "watchaudio_nondominant" in allowed_sensors else None,
    ]   
    
    # filter out None tasks
    tasks = [task for task in tasks if task is not None]
    
    # Execute tasks in parallel
    completed_tasks = []
    print(f"Starting parallel processing with {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(task): task for task in tasks}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                task_name = future.result()
                completed_tasks.append(task_name)
                print(f"Completed: {task_name}")
            except Exception as exc:
                task = future_to_task[future]
                print(f"Task {task} generated an exception: {exc}")
                import traceback
                traceback.print_exc()
    
    print(f"Parallel processing completed. All {len(completed_tasks)} tasks finished.")
    # sys.exit(1)
    # Print summary of what was processed vs skipped
    all_task_names = ['watchmotion_dominant', 'watchmotion_nondominant', 'watchaudio_dominant', 'watchaudio_nondominant']
    skipped_tasks = [task for task in all_task_names if task not in completed_tasks]
    
    if skipped_tasks:
        print(f"Skipped tasks (already existed): {skipped_tasks}")
    if completed_tasks:
        print(f"Processed tasks: {completed_tasks}")
    
    session_end_time = time.time()
    print(f"Session {session_key} processed in {session_end_time - session_start_time:.2f} seconds")
    print(f"----------------------------------------Finished processing session {session_key}----------------------------------------")
