# Extract raw window data for selected windows from session data
import os
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from datetime import datetime
import numpy as np
import argparse
import json
import sys
import pickle
import time
import shutil
import glob
import dotenv   
dotenv.load_dotenv()

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
sys.path.append(PROJECT_ROOT)

from src.featurization.doppler.DopplerFeatures import DopplerFeatures
from src.featurization.thermal.SimpleThermalFeaturizer import SimpleThermalFeaturizer
from src.featurization.depth.PersonDepthFeatures import PersonDepthFeatures
from src.featurization.pose.PoseFeatures import PoseFeatures

# Parse arguments
parser = argparse.ArgumentParser(description="Extract raw window data for selected windows.")
parser.add_argument("--participant", default="eu2a2-data-collection", help="Participant name")
parser.add_argument("--session_prefix", default="P12", help="Session prefix")
parser.add_argument("--window_size", default=5.0, help="Window size in seconds")
parser.add_argument("--sliding_window_length", default=0.5, help="Sliding window length in seconds")
parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument("--max_workers", default=10, type=int, help="Maximum number of parallel workers")
parser.add_argument("--raw_height", default=None, help="Raw height of the video")
args = parser.parse_args()

# Global config
base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")
base_results_dir = os.environ.get("BASE_RESULTS_DIR", "./results")
allowed_sensors = ["pose_depth", "persondepth_depth", "dinov2_depth", "pose_birdseye", "persondepth_birdseye", "dinov2_birdseye", "doppler", "micarray", "thermal_vision", "thermal_audio"]
# allowed_sensors = ["persondepth_depth","persondepth_birdseye"]
# Model paths
pose_model_name = "yolo11m-pose"
pose_cache_dir = os.environ.get("POSE_CACHE_DIR", f"{PROJECT_ROOT}/models/pose")
persondepth_seg_model_name = "yolo11m-seg"
persondepth_depth_model_path = os.environ.get("PERSONDEPTH_DEPTH_MODEL_PATH", f"{PROJECT_ROOT}/models/monodepth/DepthAnythingV2SmallF16.mlpackage")
persondepth_seg_cache_dir = os.environ.get("PERSONDEPTH_SEG_CACHE_DIR", f"{PROJECT_ROOT}/models/segmentation")
dinov2_cache_dir = os.environ.get("DINOV2_CACHE_DIR", f"{PROJECT_ROOT}/models/dinov2")

# Global extractors for each process
_extractors = None

def init_worker():
    """Initialize extractors once per worker process"""
    global _extractors
    print(f"[PID {os.getpid()}] Initializing...")
    _extractors = {
        'pose': PoseFeatures(),
        'thermal': SimpleThermalFeaturizer(),
        'doppler': DopplerFeatures(),
        'persondepth': PersonDepthFeatures(),
    }
    print(f"[PID {os.getpid()}] Ready")

def process_task(task_info):
    """Process a single task to extract raw window data"""
    global _extractors
    
    task_type, session_key, video_file, windows, session_output_dir, selected_windows, extra_params, raw_height = task_info
    
    print(f"[PID {os.getpid()}] {task_type} {session_key}")
    start_time = time.time()
    
    try:
        if task_type == 'pose_birdseye' or task_type == 'pose_depth':
            camera_name = "birdseye" if task_type == 'pose_birdseye' else "depth"
            _extractors['pose'].generate_raw_windows(
                video_file, windows, session_output_dir, selected_windows, camera_name,
                pose_model_name, pose_cache_dir, debug=extra_params.get('debug', False)
            )
        elif task_type == 'persondepth_depth' or task_type == 'persondepth_birdseye':
            camera_name = "birdseye" if task_type == 'persondepth_birdseye' else "depth"
            _extractors['persondepth'].generate_raw_windows(
                video_file, windows, session_output_dir, selected_windows, camera_name,
                persondepth_depth_model_path, persondepth_seg_model_name, persondepth_seg_cache_dir,
                debug=extra_params.get('debug', False), raw_height=raw_height
            )
        elif task_type == 'doppler':
            input_dir, output_dir = extra_params['input_dir'], extra_params['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            doppler_timestamps, doppler_raw_data = _extractors['doppler'].process_b64_doppler_data(input_dir, output_dir)
            if len(doppler_timestamps) > 0:
                sorted_indices = np.argsort(doppler_timestamps)
                doppler_timestamps = np.array(doppler_timestamps)[sorted_indices] / 1e9
                doppler_raw_data = np.array(doppler_raw_data)[sorted_indices]
                _extractors['doppler'].generate_raw_windows(
                    doppler_timestamps, doppler_raw_data, windows, session_output_dir, selected_windows
                )
        elif task_type == 'thermal_vision' or task_type == 'thermal_audio':
            input_dir, output_dir = extra_params['input_dir'], extra_params['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            if task_type=='thermal_vision':
                prefix="flir_vision"
            else:
                prefix="flir_audio"
            thermal_timestamps, thermal_data = _extractors['thermal'].process_b64_thermal_data(input_dir, output_dir, prefix=prefix)
            thermal_timestamps = [int(xr) for xr in thermal_timestamps]
            if len(thermal_timestamps) > 0:
                sorted_indices = np.argsort(thermal_timestamps)
                thermal_timestamps = [int(xr) for xr in thermal_timestamps]
                thermal_timestamps = np.array(thermal_timestamps)[sorted_indices] / 1e9
                thermal_data = np.array(thermal_data)[sorted_indices]
                _extractors['thermal'].generate_raw_windows(
                    thermal_timestamps, thermal_data, windows, session_output_dir, selected_windows, task_type
                )
        else:
            return f"ERROR_UNKNOWN_{task_type}_{session_key}"
        
        elapsed = time.time() - start_time
        return f"DONE_{task_type}_{session_key}_{elapsed:.1f}s"
        
    except Exception as e:
        return f"ERROR_{task_type}_{session_key}_{str(e)[:50]}"

def load_selected_windows_from_segments(segments_base_dir, participant, window_size, sliding_window_length):
    """Load selected windows from compiled segments files and return both windows and segment info"""
    selected_windows = set()
    segments_info_all = {}
    
    segments_pattern = f"{segments_base_dir}/*_segments.json"
    segment_files = glob.glob(segments_pattern)
    
    for segment_file in segment_files:
        try:
            with open(segment_file, 'r') as f:
                segments_info = json.load(f)
            segement_key = os.path.basename(segment_file).split("__")[0]
            # Extract window IDs (which are the end timestamps) and store segment info
            for window_id_str, segment_data in segments_info.items():
                window_id = float(window_id_str)
                selected_windows.add(window_id)
                segments_info_all[window_id] = segment_data
                segments_info_all[window_id]["session_key"] = segement_key
                
        except Exception as e:
            print(f"Error loading segments from {segment_file}: {e}")
    
    print(f"Loaded {len(selected_windows)} selected windows from segments")
    return list(selected_windows), segments_info_all

def copy_video_segments_to_instances(segments_base_dir, instances_base_dir, segments_info_all, selected_windows):
    """Copy video segments from segments directory to instances directory for selected windows"""
    
    copied_count = 0
    
    for window_id in selected_windows:
        if window_id in segments_info_all:

            segment_info = segments_info_all[window_id]
            session_key = segment_info["session_key"]
            start_time = datetime.fromtimestamp(segment_info["start_time"])
            end_time = datetime.fromtimestamp(segment_info["end_time"])
            
            # Create filename pattern based on how segments are named in 5c_compile_interesting_segments.py
            segment_filename = f"{start_time.strftime('%Y%m%d%H%M%S.%f')}_{end_time.strftime('%Y%m%d%H%M%S.%f')}.mp4"
            
            # Copy birdseye video if exists
            src_birdseye = f"{segments_base_dir}/birdseye/{segment_filename}"
            instance_dir = f"{instances_base_dir}/{session_key}/{window_id}"
            os.makedirs(instance_dir, exist_ok=True)
            dst_birdseye = f"{instance_dir}/birdseye.mp4"
            if os.path.exists(src_birdseye) and not os.path.exists(dst_birdseye):
                try:
                    shutil.copy2(src_birdseye, dst_birdseye)
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying birdseye segment {segment_filename}: {e}")
            
            # Copy depth video if exists
            src_depth = f"{segments_base_dir}/depth/{segment_filename}"
            dst_depth = f"{instance_dir}/depth.mp4"
            if os.path.exists(src_depth) and not os.path.exists(dst_depth):
                try:
                    shutil.copy2(src_depth, dst_depth)
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying depth segment {segment_filename}: {e}")
    
    print(f"Copied {copied_count} video segments to instances directory")

def main():
    # Get session data
    participant = args.participant
    session_prefix = args.session_prefix
    window_size = args.window_size
    sliding_window_length = args.sliding_window_length
    debug = args.debug
    max_workers = args.max_workers
    raw_height = args.raw_height
    
    # Create directories
    instances_base_dir = f"{base_results_dir}/{participant}/instances_{window_size}_{sliding_window_length}"
    segments_base_dir = f"{base_results_dir}/{participant}/segments_{window_size}_{sliding_window_length}"
    os.makedirs(instances_base_dir, exist_ok=True)
    
    # Load selected windows from segments
    selected_windows, segments_info_all = load_selected_windows_from_segments(segments_base_dir, participant, window_size, sliding_window_length)
    
    if not selected_windows:
        print("No selected windows found. Run 5c_compile_interesting_segments.py first.")
        return
    
    # Copy video segments to instances directory
    copy_video_segments_to_instances(segments_base_dir, instances_base_dir, segments_info_all, selected_windows)
    
    # Save segment metadata to instances directory
    segments_metadata_file = f"{instances_base_dir}/segments_metadata.json"
    with open(segments_metadata_file, 'w') as f:
        json.dump(segments_info_all, f, indent=4)
    print(f"Saved segment metadata to {segments_metadata_file}")
    
    base_video_dir = f"{base_processed_dir}/{participant}/processed_video_data"
    base_depth_dir = f"{base_processed_dir}/{participant}/processed_depth_data"
    base_thermal_vision_dir = f"{base_processed_dir}/{participant}/processed_thermal-vision_data"
    base_thermal_audio_dir = f"{base_processed_dir}/{participant}/processed_thermal-audio_data"
    
    # Load sessions
    watch_sessions_file = f"{base_processed_dir}/{participant}/watch_sessions_filtered.json"
    with open(watch_sessions_file, "r") as f:
        watch_sessions = json.load(f)
    
    def video_file_exists(base_filename, video_dir, depth_dir):
        return (os.path.exists(os.path.join(video_dir, base_filename)) or 
                os.path.exists(os.path.join(depth_dir, base_filename)))
    
    collected_sessions = {
        f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}": session_data
        for session_idx, session_data in enumerate(watch_sessions)
        if video_file_exists(f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}.mp4", base_video_dir, base_depth_dir)
    }
    
    print(f"Found {len(collected_sessions)} sessions")
    print(f"Selected {len(selected_windows)} windows for extraction")
    
    # Build ALL tasks for ALL sessions
    all_tasks = []
    
    for session_key, session_data in collected_sessions.items():
        # Get timestamps
        session_start = datetime.strptime(session_data['start'], "%Y%m%d_%H%M%S").timestamp()
        session_end = datetime.strptime(session_data['end'], "%Y%m%d_%H%M%S").timestamp()
        
        # Create windows
        pose_extractor = PoseFeatures()
        windows = pose_extractor.create_fixed_windows(session_start, session_end, window_size, sliding_window_length)
        
        # Create session output directory
        session_output_dir = f"{instances_base_dir}/{session_key}"
        os.makedirs(session_output_dir, exist_ok=True)
        
        # Define file paths
        birdseye_video = f"{base_video_dir}/{session_key}.mp4"
        depth_video = f"{base_depth_dir}/{session_key}.mp4"
        thermal_vision = f"{base_thermal_vision_dir}/{session_key}.mp4"
        thermal_audio = f"{base_thermal_audio_dir}/{session_key}.mp4"
        
        input_session_dir = f"{base_processed_dir}/{participant}/sessions/{'-'.join(session_key.split('-')[:2])}"
        output_session_dir = f"{base_results_dir}/{participant}/sessions/{session_key}"
        
        # Video-based tasks
        video_tasks = [
            ('pose_depth', depth_video, session_output_dir),
            ('persondepth_depth', depth_video, session_output_dir),
            ('persondepth_birdseye', birdseye_video, session_output_dir),
            ('pose_birdseye', birdseye_video, session_output_dir),
        ]
        
        for task_type, video_file, save_dir in video_tasks:
            if task_type in allowed_sensors and os.path.exists(video_file):
                task_info = (task_type, session_key, video_file, windows, save_dir, selected_windows, {'debug': debug}, raw_height)
                all_tasks.append(task_info)
        
        # Sensor data tasks
        if 'doppler' in allowed_sensors:
            doppler_task = ('doppler', session_key, '', windows, session_output_dir, selected_windows,
                          {'input_dir': input_session_dir, 'output_dir': output_session_dir, 'debug': debug}, raw_height)
            all_tasks.append(doppler_task)
            
        if 'thermal_vision' in allowed_sensors:
            thermal_vision_task = ('thermal_vision', session_key, '', windows, session_output_dir, selected_windows,
                           {'input_dir': input_session_dir, 'output_dir': output_session_dir, 'debug': debug}, raw_height)
            all_tasks.append(thermal_vision_task)

        if 'thermal_audio' in allowed_sensors:
            thermal_audio_task = ('thermal_audio', session_key, '', windows, session_output_dir, selected_windows,
                           {'input_dir': input_session_dir, 'output_dir': output_session_dir, 'debug': debug}, raw_height)
            all_tasks.append(thermal_audio_task)
    
    print(f"Total tasks: {len(all_tasks)}")
    
    # Process ALL tasks at once - no blocking between sessions
    if max_workers == 1:
        # Sequential processing
        init_worker()
        results = [process_task(task) for task in all_tasks]
    else:
        # Parallel processing - all tasks at once
        print(f"Starting {max_workers} workers for {len(all_tasks)} tasks...")
        with mp.Pool(max_workers, initializer=init_worker) as pool:
            results = pool.map(process_task, all_tasks)
    
    # Print summary
    done = [r for r in results if r.startswith('DONE_')]
    skipped = [r for r in results if r.startswith('SKIP_')]
    errors = [r for r in results if r.startswith('ERROR_')]
    
    print(f"\nSUMMARY:")
    print(f"Done: {len(done)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print(f"\nERRORS:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")

if __name__ == "__main__":
    main()