# Extract raw window data for selected windows from watch session data
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

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
sys.path.append(PROJECT_ROOT)

from src.featurization.watchmotion.MotionFeaturizer import MotionFeaturizer

# Parse arguments
parser = argparse.ArgumentParser(description="Extract raw window data for selected windows from watch data.")
parser.add_argument("--participant", default="eu2a2-data-collection", help="Participant name")
parser.add_argument("--session_prefix", default="P3", help="Session prefix")
parser.add_argument("--window_size", default=5.0, help="Window size in seconds")
parser.add_argument("--sliding_window_length", default=0.5, help="Sliding window length in seconds")
parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument("--watch_type", default="android", help="Watch type (android or iwatch)")
parser.add_argument("--max_workers", default=1, type=int, help="Maximum number of parallel workers")
parser.add_argument("--tz_rpi", default="America/New_York", help="Timezone of the RPi")
parser.add_argument("--tz_watch", default="Europe/Lisbon", help="Timezone of the watch")

args = parser.parse_args()

# Global config
base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")
base_results_dir = os.environ.get("BASE_RESULTS_DIR", "./results")
allowed_sensors = ["watchmotion_dominant", "watchmotion_nondominant", "watchaudio_dominant", "watchaudio_nondominant"]

# Thread-local storage for extractors
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
        }
    return thread_local_data.extractors

def load_selected_windows_from_segments(segments_base_dir, participant, window_size, sliding_window_length):
    """Load selected windows from compiled segments files"""
    selected_windows = set()
    
    segments_pattern = f"{segments_base_dir}/*_segments.json"
    import glob
    segment_files = glob.glob(segments_pattern)
    
    for segment_file in segment_files:
        try:
            with open(segment_file, 'r') as f:
                segments_info = json.load(f)
            
            # Extract window IDs (which are the end timestamps)
            for window_id_str in segments_info.keys():
                window_id = float(window_id_str)
                selected_windows.add(window_id)
        except Exception as e:
            print(f"Error loading segments from {segment_file}: {e}")
    
    print(f"Loaded {len(selected_windows)} selected windows from segments")
    return list(selected_windows)

def process_watch_motion_raw(input_session_dir, hand_type, windows, watch_hand_lr, debug, session_output_dir, selected_windows, watch_type, delta_seconds):
    """Process watch motion data and extract raw windows"""
    
    print(f"[Thread {threading.current_thread().name}] Processing watch-motion raw data: {input_session_dir}")
    extractors = get_extractors()
    start_time = time.time()

    # Find all watch prefix files
    watch_prefixes = glob.glob(f"{input_session_dir}/watch-{hand_type}*")
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
    
    # Process each prefix for raw data extraction
    for watch_prefix_timestamp in watch_prefix_timestamps:
        base_path = f"{input_session_dir}/watch-{hand_type}_{watch_prefix_timestamp}"
        
        # Use the generate_raw_windows function
        extractors['watch-motion'].generate_raw_windows(
            base_path, windows, session_output_dir, selected_windows, watch_hand_lr,
            watch_type=watch_type, debug=debug, delta_seconds=delta_seconds
        )
    
    end_time = time.time()
    print(f"[Thread {threading.current_thread().name}] Finished watch-motion raw data extraction in {end_time - start_time:.2f}s")
    return f'watchmotion_{hand_type}_raw'


def main():
    # Get session data
    participant = args.participant
    session_prefix = args.session_prefix
    window_size = float(args.window_size)
    sliding_window_length = float(args.sliding_window_length)
    debug = args.debug
    max_workers = args.max_workers
    watch_type = args.watch_type
    
    # Create directories
    instances_base_dir = f"{base_results_dir}/{participant}/instances_{window_size}_{sliding_window_length}"
    segments_base_dir = f"{base_results_dir}/{participant}/segments_{window_size}_{sliding_window_length}"
    os.makedirs(instances_base_dir, exist_ok=True)
    
    # Load selected windows from segments
    selected_windows = load_selected_windows_from_segments(segments_base_dir, participant, window_size, sliding_window_length)
    
    if not selected_windows:
        print("No selected windows found. Run 5c_compile_interesting_segments.py first.")
        return
    
    # Get session information
    base_video_dir = f"{base_processed_dir}/{participant}/processed_video_data"
    base_depth_dir = f"{base_processed_dir}/{participant}/processed_depth_data"
    
    watch_sessions_file = f"{base_processed_dir}/{participant}/watch_sessions_filtered.json"
    if not os.path.exists(watch_sessions_file):
        print(f"File {watch_sessions_file} does not exist. Exiting...")
        return
    
    with open(watch_sessions_file, "r") as f:
        watch_sessions = json.load(f)
    
    def video_file_exists(base_filename, video_dir, depth_dir):
        return (os.path.exists(os.path.join(video_dir, base_filename)) or 
                os.path.exists(os.path.join(depth_dir, base_filename)))
    
    collected_sessions = {
        f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}": {
            **session_data,
        }
        for session_idx, session_data in enumerate(watch_sessions)
        if video_file_exists(f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}.mp4", base_video_dir, base_depth_dir)
    }
    
    print(f"Found {len(collected_sessions)} sessions")
    print(f"Selected {len(selected_windows)} windows for raw data extraction")
    
    # Loop over sessions to process and save raw data
    for session_key in collected_sessions:
        print(f"----------------------------------------Processing session {session_key}----------------------------------------")
        session_start_time = time.time()
        
        # Get the session start and end
        session_start = collected_sessions[session_key]['start']
        session_end = collected_sessions[session_key]['end']
        session_start_timestamp = datetime.strptime(session_start, "%Y%m%d_%H%M%S").timestamp()
        session_end_timestamp = datetime.strptime(session_end, "%Y%m%d_%H%M%S").timestamp()

        # Get the windows for the session
        watch_motion_extractor = MotionFeaturizer()  # Create a temporary instance for window creation
        windows = watch_motion_extractor.create_fixed_windows(session_start_timestamp, session_end_timestamp, window_size, sliding_window_length)
        
        # Get the delta between the windows and the supposedly watch time
        delta_seconds = tz_diff(session_start.split("_")[0], args.tz_rpi, args.tz_watch)
        
        all_timestamps = [window[-1] for window in windows]
        print(f"Number of windows: {len(all_timestamps)}, first window: {all_timestamps[0]}, last window: {all_timestamps[-1]}, duration: {all_timestamps[-1] - all_timestamps[0]} seconds")
        
        # Create session output directory
        session_output_dir = f"{instances_base_dir}/{session_key}"
        os.makedirs(session_output_dir, exist_ok=True)
        
        input_session_dir = f"{base_processed_dir}/{participant}/sessions/{'-'.join(session_key.split('-')[:2])}"
        
        # check the hand for the watch from session data
        watch_hands = set()
        for watch_file in collected_sessions[session_key]['files']['non-dominant']:
            if 'watch/right' in watch_file:
                watch_hands.add('right')
            elif 'watch/left' in watch_file:
                watch_hands.add('left')
            elif 'watch/files' in watch_file:
                watch_hands.add('right')
        
        # check if there is only one value in watch_hands
        assert len(watch_hands)==1
        watch_hand_lr = list(watch_hands)[0]
        # Prepare tasks for parallel processing
        tasks = []
        
        # Motion tasks
        if "watchmotion_dominant" in allowed_sensors:
            motion_dominant_task = partial(
                process_watch_motion_raw, input_session_dir, "dominant", windows, watch_hand_lr, debug, 
                session_output_dir, selected_windows, watch_type, delta_seconds
            )
            tasks.append(motion_dominant_task)

        
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
        
        # Print summary of what was processed
        all_task_names = ['watchmotion_dominant_raw', 'watchmotion_nondominant_raw']
        skipped_tasks = [task for task in all_task_names if task not in completed_tasks]
        
        if skipped_tasks:
            print(f"Skipped tasks: {skipped_tasks}")
        if completed_tasks:
            print(f"Processed tasks: {completed_tasks}")
        
        session_end_time = time.time()
        print(f"Session {session_key} processed in {session_end_time - session_start_time:.2f} seconds")
        print(f"----------------------------------------Finished processing session {session_key}----------------------------------------")

if __name__ == "__main__":
    main()