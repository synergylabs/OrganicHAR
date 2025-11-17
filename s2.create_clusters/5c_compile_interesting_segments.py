import os
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import json
import sys
import pickle
import pytesseract
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")
base_results_dir = os.environ.get("BASE_RESULTS_DIR", "./results")

parser = argparse.ArgumentParser(description="Process video sessions.")
parser.add_argument("--participant", default="eu2a2-data-collection", help="Participant name")
parser.add_argument("--session_prefix", default="P12", help="Session prefix")
parser.add_argument("--window_size", default=5.0, help="Window size in seconds")
parser.add_argument("--sliding_window_length", default=0.5, help="Sliding window length in seconds")
parser.add_argument("--max-cluster-size", default=250, help="Maximum cluster size")
parser.add_argument("--min_confidence", default=0.9, help="Minimum confidence")
parser.add_argument("--max_samples", default=10, help="Maximum number of samples for each cluster")
parser.add_argument("--num_detections_per_min", default=1, help="Number of anomalies to be detected per minute")
parser.add_argument("--debug", action="store_true", help="Debug mode")

args = parser.parse_args()

participant = args.participant
session_prefix = args.session_prefix
window_size = args.window_size
sliding_window_length = args.sliding_window_length
max_cluster_size = args.max_cluster_size
min_confidence = args.min_confidence
max_samples = args.max_samples
num_detections_per_min = args.num_detections_per_min
debug = args.debug

# get the video and depth directories
base_video_dir = f"{base_processed_dir}/{participant}/processed_video_data"
base_depth_dir = f"{base_processed_dir}/{participant}/processed_depth_data"


# create the segments base directory
clusters_base_dir = f"{base_results_dir}/{participant}/clusters_{window_size}_{sliding_window_length}"
change_detection_base_dir = f"{base_results_dir}/{participant}/change_detection_{window_size}_{sliding_window_length}"
segments_base_dir = f"{base_results_dir}/{participant}/segments_{window_size}_{sliding_window_length}"
os.makedirs(segments_base_dir, exist_ok=True)

# Global tracking of all segments across sessions to ensure no overlap
global_segments = {}  # window_id -> segment_info

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
                "session_key":f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}",
                **session_data,
            }
            for session_idx, session_data in enumerate(watch_sessions)
            if video_file_exists(f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}.mp4",base_video_dir, base_depth_dir)
        ]
    return collected_sessions

collected_sessions = get_session_data(base_processed_dir, participant, session_prefix, window_size, sliding_window_length)

def get_non_overlapping_segments(window_ids, window_labels, window_size=5.0):
    # get the non overlapping segments
    # sort the window ids by the start time
    sorted_idx = np.argsort(window_ids)
    sorted_window_ids = np.array(window_ids)[sorted_idx]
    sorted_window_labels = np.array(window_labels)[sorted_idx]
    non_overlapping_ids, non_overlapping_labels = [], []

    for window_id, window_label in zip(sorted_window_ids, sorted_window_labels):
        if len(non_overlapping_ids) == 0:
            non_overlapping_ids.append(window_id)
            non_overlapping_labels.append(window_label)
        else:
            if window_id - non_overlapping_ids[-1] > window_size:
                non_overlapping_ids.append(window_id)
                non_overlapping_labels.append(window_label)
    return non_overlapping_ids, non_overlapping_labels

def is_segment_overlapping(start_time, end_time, existing_segments, window_size):
    """Check if a new segment overlaps with any existing segments."""
    for existing_id, existing_info in existing_segments.items():
        existing_start = existing_info["start_time"]
        existing_end = existing_info["end_time"]
        
        # Check for overlap: if segments share any time
        if not (end_time <= existing_start or start_time >= existing_end):
            return True
    return False

# loop over session to cluster and save the data
for session_index in range(1,len(collected_sessions)+1):
    current_session_key = collected_sessions[session_index-1]["session_key"]
    cluster_paths = dict(
        pose_birdseye = f"{clusters_base_dir}/{current_session_key}_pose_birdseye_clustering_results.json",
        pose_depth = f"{clusters_base_dir}/{current_session_key}_pose_depth_clustering_results.json",
        thermal_vision = f"{clusters_base_dir}/{current_session_key}_thermal_vision_clustering_results.json",
        thermal_audio = f"{clusters_base_dir}/{current_session_key}_thermal_audio_clustering_results.json",
        doppler = f"{clusters_base_dir}/{current_session_key}_doppler_clustering_results.json",
        micarray = f"{clusters_base_dir}/{current_session_key}_micarray_clustering_results.json",
        persondepth_birdseye = f"{clusters_base_dir}/{current_session_key}_persondepth_birdseye_clustering_results.json",
        persondepth_depth = f"{clusters_base_dir}/{current_session_key}_persondepth_depth_clustering_results.json",
        dinov2_birdseye = f"{clusters_base_dir}/{current_session_key}_dinov2_birdseye_clustering_results.json",
        dinov2_depth = f"{clusters_base_dir}/{current_session_key}_dinov2_depth_clustering_results.json",
    )

    change_detection_paths = dict(
        pose_birdseye = f"{change_detection_base_dir}/{current_session_key}_pose_birdseye_top_{num_detections_per_min}.csv",
        pose_depth = f"{change_detection_base_dir}/{current_session_key}_pose_depth_top_{num_detections_per_min}.csv",
        thermal_vision = f"{change_detection_base_dir}/{current_session_key}_thermal_vision_top_{num_detections_per_min}.csv",
        doppler = f"{change_detection_base_dir}/{current_session_key}_doppler_top_{num_detections_per_min}.csv",
        micarray = f"{change_detection_base_dir}/{current_session_key}_micarray_top_{num_detections_per_min}.csv",
        persondepth_birdseye = f"{change_detection_base_dir}/{current_session_key}_persondepth_birdseye_top_{num_detections_per_min}.csv",
        persondepth_depth = f"{change_detection_base_dir}/{current_session_key}_persondepth_depth_top_{num_detections_per_min}.csv",
        dinov2_birdseye = f"{change_detection_base_dir}/{current_session_key}_dinov2_birdseye_top_{num_detections_per_min}.csv",
        dinov2_depth = f"{change_detection_base_dir}/{current_session_key}_dinov2_depth_top_{num_detections_per_min}.csv",
    )

    # extract the segments from the clusters
    segments_file = f"{segments_base_dir}/{current_session_key}__maxsize-{max_cluster_size}_conf-{min_confidence}_maxsamples-{max_samples}_segments.json"
    if os.path.exists(segments_file):
        segments_info = json.load(open(segments_file, "r"))
    else:
        segments_info = dict()
        # loop over the clusters and extract the segments
        for sensor_name, cluster_path in cluster_paths.items():
            if not os.path.exists(cluster_path):
                print(f"Cluster file {cluster_path} does not exist. Skipping...")
                continue
            with open(cluster_path, "r") as f:
                try:
                    cluster_data = json.load(f)
                except Exception as e:
                    print(f"Error loading cluster data from {cluster_path}. Skipping...")
                    print(f"Error: {e}")
                    continue
                window_ids = cluster_data["window_ids"]
                labels = cluster_data["labels"]
                confidences = np.array(cluster_data["confidences"])
                unique_labels, label_counts = np.unique(labels, return_counts=True)
                # select labels that are greater than 1 and have total count less than 100
                filtered_labels = unique_labels[np.where((label_counts > 1) & (label_counts < max_cluster_size))[0]]
                filtered_window_idxs = np.where(np.isin(labels, filtered_labels) & (confidences > min_confidence) & (labels != -1))[0]
                # get the window ids, and labels for the eligible window idxs
                filtered_window_ids = np.array(window_ids)[filtered_window_idxs]
                filtered_window_labels = np.array(labels)[filtered_window_idxs]
                # add the sensor name to the labels
                filtered_window_labels = np.array([f"{sensor_name}_{label}" for label in filtered_window_labels])
                filtered_confidences = np.array(confidences)[filtered_window_idxs]

                # get the non overlapping segments
                non_overlapping_ids, non_overlapping_labels = get_non_overlapping_segments(filtered_window_ids, filtered_window_labels, window_size=window_size)
                
                # create label dict with label as key and list of window ids as value
                label_dict = dict()
                for window_id, window_label in zip(non_overlapping_ids, non_overlapping_labels):
                    if window_label not in label_dict:
                        label_dict[window_label] = []
                    label_dict[window_label].append(window_id)
                # sample max_samples from the label dict
                selected_label_dict={}
                for label, label_window_ids in label_dict.items():
                    if len(label_window_ids) > max_samples:
                        selected_window_ids = np.random.choice(label_window_ids, size=max_samples, replace=False)
                        selected_label_dict[label] = selected_window_ids
                        for window_id in selected_window_ids:
                            start_time = float(window_id) - window_size
                            end_time = float(window_id)
                            
                            # Check if this segment overlaps with any existing segments across all sessions
                            if not is_segment_overlapping(start_time, end_time, global_segments, window_size):
                                if window_id not in segments_info:
                                    segments_info[float(window_id)] = dict(
                                        window_id=float(window_id),
                                        end_time=end_time,
                                        start_time=start_time,
                                        labels=[str(label)], # list of labels for the window
                                    )
                                    # Add to global tracking
                                    global_segments[float(window_id)] = segments_info[float(window_id)]
                                else:
                                    segments_info[float(window_id)]["labels"].append(str(label))
                            # If overlapping, discard the segment
                    else:
                        for window_id in label_window_ids:
                            start_time = float(window_id) - window_size
                            end_time = float(window_id)
                            
                            # Check if this segment overlaps with any existing segments across all sessions
                            if not is_segment_overlapping(start_time, end_time, global_segments, window_size):
                                if window_id not in segments_info:
                                    segments_info[float(window_id)] = dict(
                                        window_id=float(window_id),
                                        end_time=end_time,
                                        start_time=start_time,
                                        labels=[str(label)],
                                    )
                                    # Add to global tracking
                                    global_segments[float(window_id)] = segments_info[float(window_id)]
                                else:
                                    segments_info[float(window_id)]["labels"].append(str(label))
                            # If overlapping, discard the segment
            clustering_window_starts = np.array([segments_info[window_id]["start_time"] for window_id in segments_info.keys()])
            clustering_window_ends = np.array([segments_info[window_id]["end_time"] for window_id in segments_info.keys()])
            # loop over the change detection paths and add the change detection results to the segments info
            for sensor_name, change_detection_path in change_detection_paths.items():
                if not os.path.exists(change_detection_path):
                    print(f"Change detection file {change_detection_path} does not exist. Skipping...")
                    continue
                df_change_detection = pd.read_csv(change_detection_path, index_col=0)
                change_timestamps = df_change_detection.index.values
                for change_timestamp in change_timestamps:
                    # check if this is already in one of the windows in segments_info
                    change_timestamp_idx = np.where((clustering_window_starts <= change_timestamp) & (clustering_window_ends > change_timestamp))[0]
                    if len(change_timestamp_idx) > 0:
                        change_window_id = clustering_window_ends[change_timestamp_idx[0]]
                        segments_info[float(change_window_id)]["labels"].append(f"{sensor_name}_anomaly")
                    else:
                        # create a new window
                        new_window_id = int(change_timestamp)
                        new_window_start = int(change_timestamp - window_size)
                        new_window_end = int(change_timestamp)
                        new_window_labels = [f"{sensor_name}_anomaly"]
                        
                        # Check if this new segment overlaps with any existing segments across all sessions
                        if not is_segment_overlapping(new_window_start, new_window_end, global_segments, window_size):
                            segments_info[float(new_window_id)] = dict(
                                window_id=float(new_window_id),
                                end_time = new_window_end,
                                start_time = new_window_start,
                                labels=new_window_labels,
                            )
                            # Add to global tracking
                            global_segments[float(new_window_id)] = segments_info[float(new_window_id)]
                        # If overlapping, discard the segment
                    # print(f"[{current_session_key}] Added anomaly for {sensor_name} at {change_timestamp}")


        
        # save the segments info to a json file
        with open(segments_file, "w") as f:
            json.dump(segments_info, f, indent=4)
        
        # Print summary of segments across all sessions
        print(f"[{current_session_key}] Session segments: {len(segments_info)}")
        print(f"[{current_session_key}] Total segments across all sessions: {len(global_segments)}")

    # now we wish to create the segments from the files based on the segments
    # sort the segments info by the start time
    segments_info_list = sorted([x for x in segments_info.values()], key=lambda x: x["start_time"])
    segment_sessions = []
    for segment_info in segments_info_list:
        segment_end_time = segment_info["end_time"]
        # find the collected session that has lower start time and higher end time
        session_key = None
        for collected_session in collected_sessions:
            collected_session_start_timestamp = datetime.fromisoformat(collected_session["start_time"]).timestamp()
            collected_session_end_timestamp = datetime.fromisoformat(collected_session["end_time"]).timestamp()
            if collected_session_start_timestamp <= segment_end_time and collected_session_end_timestamp >= segment_info["start_time"]:
                session_key = collected_session["session_key"]
                break
        if session_key is None:
            session_key = current_session_key
        segment_sessions.append(session_key)


    segment_birdseye_paths = [f"{base_video_dir}/{session_key}.mp4" for session_key in segment_sessions]
    segment_depth_paths = [f"{base_depth_dir}/{session_key}.mp4" for session_key in segment_sessions]
    

    # create the output directory
    segment_output_dir = f"{segments_base_dir}"
    os.makedirs(f"{segment_output_dir}/birdseye", exist_ok=True)
    os.makedirs(f"{segment_output_dir}/depth", exist_ok=True)

    # loop over the segments and extract the videos
    for segment_idx in range(len(segments_info_list)):
        segment_info = segments_info_list[segment_idx]
        segment_start_time = datetime.fromtimestamp(segment_info["start_time"])
        segment_end_time = datetime.fromtimestamp(segment_info["end_time"])
        segment_session_key = segment_sessions[segment_idx]
        birseye_video_path = segment_birdseye_paths[segment_idx]
        depth_video_path = segment_depth_paths[segment_idx]
        birdseye_segment_output_path = f"{segment_output_dir}/birdseye/{segment_start_time.strftime('%Y%m%d%H%M%S.%f')}_{segment_end_time.strftime('%Y%m%d%H%M%S.%f')}.mp4"
        depth_segment_output_path = f"{segment_output_dir}/depth/{segment_start_time.strftime('%Y%m%d%H%M%S.%f')}_{segment_end_time.strftime('%Y%m%d%H%M%S.%f')}.mp4"
        if not os.path.exists(birdseye_segment_output_path):
            extract_segment_video(birseye_video_path, segment_start_time, segment_end_time, birdseye_segment_output_path)
        if not os.path.exists(depth_segment_output_path):
            extract_segment_video(depth_video_path, segment_start_time, segment_end_time, depth_segment_output_path)

        print(f"[{segment_session_key}] Extracted segment {segment_idx}/{len(segments_info_list)} - ({segment_start_time} - {segment_end_time})")

