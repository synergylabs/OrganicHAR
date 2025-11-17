import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import argparse
import json
import sys
import random
import dotenv
dotenv.load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")
base_results_dir = os.environ.get("BASE_RESULTS_DIR", "./results")

parser = argparse.ArgumentParser(description="Randomly extract ground truth segments from video sessions for AV activity prediction.")
parser.add_argument("--participant", default="eu2a2-data-collection", help="Participant name")
parser.add_argument("--session_prefix", default="P12", help="Session prefix")
parser.add_argument("--window_size", default=5.0, help="Window size in seconds")
parser.add_argument("--sliding_window_length", default=0.5, help="Sliding window length in seconds")
parser.add_argument("--description_vlm_model", default="gpt-4.1", help="VLM model used for description generation")
parser.add_argument("--clustering_vlm_model", default="gpt-4.1", help="VLM model used for location and activity clustering")
parser.add_argument("--merging_threshold", default='auto', help="Threshold for merging similar labels")
parser.add_argument("--direction", default="forward", choices=["forward", "backward"], help="Selection direction")
parser.add_argument("--early_sessions", default="1-2", help="Range of early session indices (e.g., '1-3')")
parser.add_argument("--middle_sessions", default="3-4", help="Range of middle session indices (e.g., '4-6')")
parser.add_argument("--late_sessions", default="5-10", help="Range of late session indices (e.g., '7-10')")
parser.add_argument("--early_samples", default=200, type=int, help="Number of samples from early sessions")
parser.add_argument("--middle_samples", default=200, type=int, help="Number of samples from middle sessions")
parser.add_argument("--late_samples", default=200, type=int, help="Number of samples from late sessions")
parser.add_argument("--min_session_duration", default=30.0, help="Minimum session duration in seconds to consider")
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
parser.add_argument("--debug", action="store_true", help="Debug mode")

args = parser.parse_args()

participant = args.participant
session_prefix = args.session_prefix
window_size = args.window_size
sliding_window_length = args.sliding_window_length
description_vlm_model = args.description_vlm_model
clustering_vlm_model = args.clustering_vlm_model
merging_threshold = args.merging_threshold
direction = args.direction
early_sessions = args.early_sessions
middle_sessions = args.middle_sessions
late_sessions = args.late_sessions
early_samples = args.early_samples
middle_samples = args.middle_samples
late_samples = args.late_samples
min_session_duration = args.min_session_duration
seed = args.seed
debug = args.debug

# Set random seed for reproducibility
random.seed(seed)
np.random.seed(seed)

# get the video and depth directories
base_video_dir = f"{base_processed_dir}/{participant}/processed_video_data"
base_depth_dir = f"{base_processed_dir}/{participant}/processed_depth_data"

# create the GT segments base directory
gt_segments_base_dir = f"{base_results_dir}/{participant}/gt_segments"
label_generation_base_dir = f"{base_results_dir}/{participant}/leaveoneout_analysis/label_generation_{window_size}_{sliding_window_length}/{description_vlm_model}"
activity_predictions_base_dir = f"{base_results_dir}/{participant}/leaveoneout_analysis/activity_predictions_{window_size}_{sliding_window_length}/{description_vlm_model}"
os.makedirs(gt_segments_base_dir, exist_ok=True)

def parse_filename_timestamps(filename: str):
    """Parse start and end timestamps from filename."""
    import os
    import re
    from datetime import datetime
    basename = os.path.basename(filename)
    pattern = r'.*-(\d{8}_\d{6})_(\d{8}_\d{6})\.mp4$'
    match = re.search(pattern, basename)
    if not match:
        raise ValueError(f"Filename {basename} doesn't match expected pattern: <prefix>_<start_time>_<end_time>.mp4")
    start_time_str, end_time_str = match.groups()
    start_time = datetime.strptime(start_time_str, '%Y%m%d_%H%M%S')
    end_time = datetime.strptime(end_time_str, '%Y%m%d_%H%M%S')
    return start_time, end_time

def extract_segment_video(video_path, start_time, end_time, output_path):
    """Extract video segment using ffmpeg."""
    # extract the segment video
    video_start_time, video_end_time = parse_filename_timestamps(video_path)
   
    # get the video duration
    video_duration = video_end_time - video_start_time
    
    # get the segment duration
    segment_duration = end_time - start_time
    
    # get the start and end times for the segment
    segment_start_time = start_time - video_start_time
    segment_end_time = min(segment_start_time + segment_duration, video_duration)
    
    # use ffmpeg to extract the segment
    try:
        os.system(f"ffmpeg -i {video_path} -ss {str(segment_start_time)} -to {str(segment_end_time)} -c copy {output_path}")
        return True
    except Exception as e:
        print(f"Error extracting segment video: {e}, video_path: {video_path}, start_time: {start_time}, end_time: {end_time}, output_path: {output_path}")
        return False

def get_session_data(base_processed_dir, participant, session_prefix, window_size, sliding_window_length):
    """Get all session information."""
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

    def video_file_exists(base_filename, video_dir, depth_dir):
        # check if the file exists in the directory
        if os.path.exists(os.path.join(video_dir, base_filename)):
            return True
        if os.path.exists(os.path.join(depth_dir, base_filename)):
            return True
        return False

    collected_sessions = [
        {
            "session_key": f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}",
            **session_data,
        }
        for session_idx, session_data in enumerate(watch_sessions)
        if video_file_exists(f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}.mp4", base_video_dir, base_depth_dir)
    ]
    for session_idx in range(len(collected_sessions)-1):
        collected_sessions[session_idx]["predicted_session_key"] = collected_sessions[session_idx+1]["session_key"]
    collected_sessions = collected_sessions[:-1]
    return collected_sessions

def get_session_duration(session_data):
    """Calculate session duration in seconds."""
    start_time = datetime.fromisoformat(session_data["start_time"])
    end_time = datetime.fromisoformat(session_data["end_time"])
    return (end_time - start_time).total_seconds()

def load_ensemble_predictions(activity_predictions_base_dir, session_key):
    """Load ensemble predictions for a specific session."""
    ensemble_file = f"{activity_predictions_base_dir}/{session_key}/ensemble_predictions/predictions.csv"
    
    if not os.path.exists(ensemble_file):
        print(f"Ensemble predictions file not found: {ensemble_file}")
        return pd.DataFrame()
    
    try:
        df_predictions = pd.read_csv(ensemble_file)
        return df_predictions
    except Exception as e:
        print(f"Error loading ensemble predictions from {ensemble_file}: {e}")
        return pd.DataFrame()

def parse_session_range(range_str):
    """Parse session range string like '1-3' into list [1, 2, 3]."""
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(range_str)]

def extract_session_index(session_key, session_prefix):
    """Extract session index from session key like 'P3-01-...' -> 1."""
    try:
        # Extract the session number after the prefix
        parts = session_key.split('-')
        if len(parts) >= 2:
            session_num = int(parts[1])
            return session_num
        return None
    except (ValueError, IndexError):
        return None

def group_sessions_by_temporal_stage(collected_sessions, session_prefix, early_range, middle_range, late_range):
    """Group sessions into early, middle, and late temporal stages."""
    early_indices = parse_session_range(early_range)
    middle_indices = parse_session_range(middle_range)
    late_indices = parse_session_range(late_range)
    
    grouped_sessions = {
        'early': [],
        'middle': [],
        'late': [],
        'unclassified': []
    }
    
    for session_idx, session in enumerate(collected_sessions):
        # session_idx = extract_session_index(session['session_key'], session_prefix)
        
        if session_idx is None:
            grouped_sessions['unclassified'].append(session)
        elif session_idx+1 in early_indices:
            grouped_sessions['early'].append(session)
        elif session_idx+1 in middle_indices:
            grouped_sessions['middle'].append(session)
        elif session_idx+1 in late_indices:
            grouped_sessions['late'].append(session)
        else:
            grouped_sessions['unclassified'].append(session)
    
    return grouped_sessions

def sample_from_session_group(sessions, num_samples, group_name, window_size, min_session_duration, activity_predictions_base_dir, sample_id_counter, df_location_action):
    """Sample specified number of segments from a group of sessions."""
    valid_sessions = [s for s in sessions if get_session_duration(s) >= min_session_duration]
    if not valid_sessions:
        print(f"No valid sessions in {group_name} group (minimum duration: {min_session_duration}s)")
        return [], sample_id_counter
    
    # Collect all available timestamps from valid sessions
    all_session_timestamps = []
    
    for session in valid_sessions:
        session_key = session["session_key"]
        
        # get the location action labels for this session
        df_session_location_action = df_location_action[df_location_action['session_key'] == session_key]
        if df_session_location_action.empty:
            if debug:
                print(f"No location action labels found for session {session_key} in {group_name} group")
            continue
        
        # Add session info to each timestamp
        for _, row in df_session_location_action.iterrows():
            all_session_timestamps.append({
                'predicted_session_key': session_key,
                'window_id': row['window_id'],
                'prediction': "",
                'confidence': 0.0,
                'gt_label': "",
                'location_cluster': row['location_cluster'],
                'activity_cluster': row['activity_cluster']
            })
    
    if not all_session_timestamps:
        print(f"No prediction timestamps available for {group_name} group")
        return [], sample_id_counter
    
    # Sample the requested number (or all available if fewer)
    actual_samples = min(num_samples, len(all_session_timestamps))
    selected_indices = np.random.choice(len(all_session_timestamps), size=actual_samples, replace=False)
    
    samples = []
    for idx in selected_indices:
        timestamp_info = all_session_timestamps[idx]
        
        # window_id is the end timestamp
        end_time = datetime.fromtimestamp(timestamp_info['window_id'])
        start_time = end_time - timedelta(seconds=window_size)
        
        sample_info = {
            "sample_id": sample_id_counter,
            "predicted_session_key": timestamp_info['predicted_session_key'],
            "start_time": start_time,
            "end_time": end_time,
            "window_id": timestamp_info['window_id'],
            "prediction": timestamp_info['prediction'],
            "confidence": timestamp_info['confidence'],
            "gt_label": timestamp_info['gt_label'],
            "location_cluster": timestamp_info['location_cluster'],
            "activity_cluster": timestamp_info['activity_cluster'],
            "temporal_group": group_name,
        }
        
        samples.append(sample_info)
        sample_id_counter += 1
        
        if debug:
            print(f"Selected {group_name} sample {sample_id_counter-1}: {timestamp_info['session_key']} at {start_time} -> {end_time} (pred: {timestamp_info['prediction']}, conf: {timestamp_info['confidence']:.3f})")
    
    print(f"{group_name.capitalize()} group: {actual_samples}/{num_samples} samples from {len(all_session_timestamps)} available timestamps across {len(valid_sessions)} sessions")
    
    return samples, sample_id_counter

def generate_temporal_stratified_samples(collected_sessions, session_prefix, early_range, middle_range, late_range, 
                                        early_samples, middle_samples, late_samples, window_size, min_session_duration, activity_predictions_base_dir):
    """Generate samples using temporal stratified sampling strategy."""
    
    # Group sessions by temporal stage
    grouped_sessions = group_sessions_by_temporal_stage(collected_sessions, session_prefix, early_range, middle_range, late_range)

    # for collected sessions, get the mapping between window_id and session_key
    window_id_to_session_key = {}
    for session in collected_sessions:
        mappings_file = f"{base_results_dir}/{participant}/segments_{window_size}_{sliding_window_length}/{session['session_key']}__maxsize-100_conf-0.95_maxsamples-10_segments.json"
        if os.path.exists(mappings_file):
            with open(mappings_file, "r") as f:
                mappings = json.load(f)
                for mapping in mappings:
                    window_id_to_session_key[mapping] = session['session_key']

    # extract the location action labels for each session
    label_generation_base_dir = f"{base_results_dir}/{participant}/leaveoneout_analysis/label_generation_{window_size}_{sliding_window_length}/{description_vlm_model}"
    location_action_labels = []
    for session in collected_sessions:
        session_key = session["session_key"]
        location_action_file = f"{label_generation_base_dir}/{session_key}/location_action_labels.csv"
        if os.path.exists(location_action_file):
            df_location_action = pd.read_csv(location_action_file)
            df_location_action['session_key'] = df_location_action['window_id'].astype(str).map(window_id_to_session_key)
            location_action_labels.append(df_location_action)

    df_location_action = pd.concat(location_action_labels)
    df_location_action = df_location_action.drop_duplicates()
    df_location_action = df_location_action[df_location_action['activity_cluster'] != 'no_match'] # remove the no_match activity cluster
    df_location_action = df_location_action[df_location_action['location_cluster'] != 'no_match'] # remove the no_match location cluster
    df_location_action = df_location_action[df_location_action['location_cluster'] != ''] # remove the empty location cluster
    df_location_action = df_location_action[df_location_action['location_cluster'] != 'nan'] # remove the nan location cluster
    df_location_action = df_location_action[df_location_action['activity_cluster'] != ''] # remove the empty activity cluster
    df_location_action = df_location_action[df_location_action['activity_cluster'] != 'nan'] # remove the nan activity cluster
    df_location_action = df_location_action[df_location_action['location_cluster'] != 'unknown'] # remove the unknown location cluster
    df_location_action = df_location_action[df_location_action['activity_cluster'] != 'unknown'] # remove the unknown activity cluster
    df_location_action = df_location_action[df_location_action['location_cluster'] != 'unknown_location'] # remove the unknown_location location cluster
    df_location_action = df_location_action[df_location_action['activity_cluster'] != 'unknown_activity'] # remove the unknown_activity activity cluster
    df_location_action = df_location_action[df_location_action['location_cluster'] != 'unknown_location'] # remove the unknown_location location cluster


    
    print(f"Session grouping:")
    print(f"  Early ({early_range}): {len(grouped_sessions['early'])} sessions")
    print(f"  Middle ({middle_range}): {len(grouped_sessions['middle'])} sessions") 
    print(f"  Late ({late_range}): {len(grouped_sessions['late'])} sessions")
    print(f"  Unclassified: {len(grouped_sessions['unclassified'])} sessions")
    
    all_samples = []
    sample_id_counter = 250 # start from 250 to avoid conflicts with the leaveoneout analysis
    
    # Sample from each group
    early_samples_list, sample_id_counter = sample_from_session_group(
        grouped_sessions['early'], early_samples, 'early', window_size, min_session_duration, activity_predictions_base_dir, sample_id_counter, df_location_action
    )
    all_samples.extend(early_samples_list)
    
    middle_samples_list, sample_id_counter = sample_from_session_group(
        grouped_sessions['middle'], middle_samples, 'middle', window_size, min_session_duration, activity_predictions_base_dir, sample_id_counter, df_location_action
    )
    all_samples.extend(middle_samples_list)
    
    late_samples_list, sample_id_counter = sample_from_session_group(
        grouped_sessions['late'], late_samples, 'late', window_size, min_session_duration, activity_predictions_base_dir, sample_id_counter, df_location_action
    )
    all_samples.extend(late_samples_list)
    
    # Handle unclassified sessions if any
    if grouped_sessions['unclassified']:
        print(f"Warning: {len(grouped_sessions['unclassified'])} sessions could not be classified into temporal groups")
        for session in grouped_sessions['unclassified']:
            print(f"  Unclassified: {session['session_key']}")
    
    return all_samples

# Get session data
collected_sessions = get_session_data(base_processed_dir, participant, session_prefix, window_size, sliding_window_length)
print(f"Found {len(collected_sessions)} total sessions")

# Generate samples using temporal stratified sampling
prediction_samples = generate_temporal_stratified_samples(
    collected_sessions, session_prefix, early_sessions, middle_sessions, late_sessions,
    early_samples, middle_samples, late_samples, window_size, min_session_duration, activity_predictions_base_dir
)
total_target_samples = early_samples + middle_samples + late_samples
print(f"Generated {len(prediction_samples)}/{total_target_samples} samples using temporal stratified sampling")

if not prediction_samples:
    print("No samples generated. Exiting...")
    exit(1)

# Create output directories
total_target_samples = early_samples + middle_samples + late_samples
gt_output_dir = f"{gt_segments_base_dir}/temporal_stratified_250_samples"
os.makedirs(f"{gt_output_dir}/birdseye", exist_ok=True)
os.makedirs(f"{gt_output_dir}/depth", exist_ok=True)

# Save sample metadata
samples_metadata = {
    "participant": participant,
    "session_prefix": session_prefix,
    "window_size": window_size,
    "sliding_window_length": sliding_window_length,
    "description_vlm_model": description_vlm_model,
    "clustering_vlm_model": clustering_vlm_model,
    "merging_threshold": merging_threshold,
    "direction": direction,
    "sampling_strategy": "temporal_stratified",
    "early_sessions": early_sessions,
    "middle_sessions": middle_sessions,
    "late_sessions": late_sessions,
    "early_samples": early_samples,
    "middle_samples": middle_samples,
    "late_samples": late_samples,
    "total_target_samples": total_target_samples,
    "min_session_duration": min_session_duration,
    "seed": seed,
    "generated_samples": len(prediction_samples),
    "samples": [
        {
            "sample_id": sample["sample_id"],
            "predicted_session_key": sample["predicted_session_key"],
            "start_time": sample["start_time"].isoformat(),
            "end_time": sample["end_time"].isoformat(),
            "window_id": sample["window_id"],
            "prediction": sample["prediction"],
            "confidence": sample["confidence"],
            "gt_label": sample["gt_label"],
            "temporal_group": sample["temporal_group"],
            "location_cluster": sample["location_cluster"],
            "activity_cluster": sample["activity_cluster"],
        }
        for sample in prediction_samples
    ]
}

existing_metadata_file = f"{gt_output_dir}/samples_metadata.json"
# load the exit
if os.path.exists(existing_metadata_file):
    with open(existing_metadata_file, "r") as f:
        existing_metadata = json.load(f)
    samples_metadata["samples"] = existing_metadata["samples"] + samples_metadata["samples"]
    samples_metadata['generated_samples'] = len(samples_metadata['samples']) + existing_metadata['generated_samples']

metadata_file = f"{gt_output_dir}/samples_metadata_av.json"
with open(metadata_file, "w") as f:
    json.dump(samples_metadata, f, indent=4)

print(f"Saved metadata to {metadata_file}")

# Extract video segments
successful_extractions = 0
failed_extractions = 0

for sample in prediction_samples:
    sample_id = sample["sample_id"]
    start_time = sample["start_time"]
    end_time = sample["end_time"]
    prediction_session_key = sample["predicted_session_key"]
    prediction = sample["prediction"]
    confidence = sample["confidence"]
    gt_label = sample["gt_label"]
    temporal_group = sample["temporal_group"]
    
    # Construct video paths
    birdseye_video_path = f"{base_video_dir}/{prediction_session_key}.mp4"
    depth_video_path = f"{base_depth_dir}/{prediction_session_key}.mp4"
    
    # Construct output paths with prediction info
    # Clean prediction and gt_label to remove special characters for filenames
    clean_prediction = prediction.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if type(gt_label) == str:
        clean_gt_label = gt_label.replace(" ", "_").replace("/", "_").replace("\\", "_") if gt_label else ""
    else:
        clean_gt_label = ""
    
    sample_filename = f"sample--{str(sample_id).zfill(3)}--{temporal_group}--{end_time.strftime('%Y%m%d_%H%M%S')}"
    if clean_gt_label:
        sample_filename += f"_gt-{clean_gt_label}"
    
    birdseye_output_path = f"{gt_output_dir}/birdseye/{sample_filename}.mp4"
    depth_output_path = f"{gt_output_dir}/depth/{sample_filename}.mp4"
    
    # Extract birdseye video
    if os.path.exists(birdseye_video_path):
        if not os.path.exists(birdseye_output_path):
            if extract_segment_video(birdseye_video_path, start_time, end_time, birdseye_output_path):
                successful_extractions += 1
                if debug:
                    print(f"✓ Extracted birdseye sample {sample_id}: {sample_filename}")
            else:
                failed_extractions += 1
                print(f"✗ Failed to extract birdseye sample {sample_id}: {sample_filename}")
        else:
            if debug:
                print(f"- Birdseye sample {sample_id} already exists: {sample_filename}")
    else:
        print(f"⚠ Birdseye video not found: {birdseye_video_path}")
        failed_extractions += 1
    
    # Extract depth video
    # if os.path.exists(depth_video_path):
    #     if not os.path.exists(depth_output_path):
    #         if extract_segment_video(depth_video_path, start_time, end_time, depth_output_path):
    #             if debug:
    #                 print(f"✓ Extracted depth sample {sample_id}: {sample_filename}")
    #         else:
    #             print(f"✗ Failed to extract depth sample {sample_id}: {sample_filename}")
    #     else:
    #         if debug:
    #             print(f"- Depth sample {sample_id} already exists: {sample_filename}")
    # else:
    #     print(f"⚠ Depth video not found: {depth_video_path}")
    
    print(f"[{sample_id+1}/{len(prediction_samples)}] Processed {temporal_group} sample from {prediction_session_key}: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')} (pred: {prediction}, conf: {confidence:.3f})")

print(f"\n=== Extraction Complete ===")
print(f"Total samples processed: {len(prediction_samples)}/{total_target_samples}")

# Count samples by temporal group
group_counts = {}
for sample in prediction_samples:
    group = sample['temporal_group']
    group_counts[group] = group_counts.get(group, 0) + 1

print(f"Sample distribution:")
print(f"  Early ({early_sessions}): {group_counts.get('early', 0)}/{early_samples}")
print(f"  Middle ({middle_sessions}): {group_counts.get('middle', 0)}/{middle_samples}")
print(f"  Late ({late_sessions}): {group_counts.get('late', 0)}/{late_samples}")

print(f"Successful extractions: {successful_extractions}")
print(f"Failed extractions: {failed_extractions}")
print(f"Output directory: {gt_output_dir}")
print(f"Metadata file: {metadata_file}")
