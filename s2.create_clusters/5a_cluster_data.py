import os
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import json
import traceback
import sys
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from common.clustering.hdbscanClustering import HDBSCANClustering, FeatureResults, SensorClusteringConfig, ClusteringResults


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
watch_features_base_dir = f"{base_results_dir}/{participant}/features_{window_size}_{sliding_window_length}"
clusters_base_dir = f"{base_results_dir}/{participant}/clusters_{window_size}_{sliding_window_length}"
os.makedirs(clusters_base_dir, exist_ok=True)

allowed_sensors = ["pose_depth","persondepth_depth","dinov2_depth","pose_birdseye","persondepth_birdseye","dinov2_birdseye", "watchmotion_dominant", "watchaudio_dominant"]
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
for session_index in range(1,len(collected_sessions)+1):
    clustering_sessions = {x["session_key"]: x for x in collected_sessions[:session_index]}
    clustering_session_keys = [x["session_key"] for x in collected_sessions[:session_index]]
    print(f"Clustering sessions: {clustering_session_keys}")

    # get the session data from all the sessions
    all_feature_results = {}
    all_sensor_names = set()

    # get the feature results from all the sessions
    for session_key in clustering_sessions:
        session_data = clustering_sessions[session_key]
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
            watchmotion_dominant = f"{watch_features_base_dir}/{session_key}_watch_motion.pkl" if 'watchmotion_dominant' in allowed_sensors else None,
            watchmotion_nondominant = f"{watch_features_base_dir}/{session_key}_watch_motion.pkl" if 'watchmotion_nondominant' in allowed_sensors else None,
            watchaudio_dominant = f"{watch_features_base_dir}/{session_key}_watch_audio.pkl" if 'watchaudio_dominant' in allowed_sensors else None,
            watchaudio_nondominant = f"{watch_features_base_dir}/{session_key}_watch_audio.pkl" if 'watchaudio_nondominant' in allowed_sensors else None,
        )
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
            sensor_data['features'] = [np.array(x).flatten() for x in sensor_data['features'] if x is not None]
            sensor_data['timestamps'] = [x for x, y in zip(sensor_data['timestamps'], none_mask) if y]
            df_sensor_data = pd.DataFrame.from_records(sensor_data['features'], index=sensor_data['timestamps'])
            if sensor_name not in all_feature_results:
                all_feature_results[sensor_name] = df_sensor_data
                all_sensor_names.add(sensor_name)
            else:
                all_feature_results[sensor_name] = pd.concat([all_feature_results[sensor_name], df_sensor_data], axis=0)


    for sensor_name in all_sensor_names:
        print(f"Clustering {sensor_name}...")
        sensor_feature_results = FeatureResults(
            features=all_feature_results[sensor_name].values,
            feature_names=all_feature_results[sensor_name].columns.tolist(),
            window_ids=all_feature_results[sensor_name].index.values
        )
        # cluster the data
        clustering_config = SensorClusteringConfig(
            n_components_range=[8, 12, 15],
            min_cluster_size_range=[5, 8],
            min_samples_range=[3, 5, 8],
            min_desired_clusters=5,
            max_desired_clusters=40,
            max_noise_ratio=0.5,
            min_cluster_size=5
        )
        clusterer = HDBSCANClustering(write_dir=clusters_base_dir, config=clustering_config)
        clustering_results_path = f"{clusters_base_dir}/{session_key}_{sensor_name}_clustering_results.json"
        if not os.path.exists(clustering_results_path):
            try:
                clustering_results = clusterer.fit(sensor_feature_results)
            except Exception as e:
                print(f"Unable to cluster for {clustering_results_path}, {e}")
                continue
            # get a summary of the clustering results
            print(f"Clustering results for {session_key}-{sensor_name}: {clustering_results_path}")
            print(f"Number of clusters: {np.unique(clustering_results.labels).shape[0]}")
            # clusterer.save(clustering_results_path)
            # save the clustering results to a json file
            with open(clustering_results_path, "w") as f:
                json.dump(clustering_results.to_dict(), f, indent=4)
        else:
            print(f"Clustering results for {session_key}-{sensor_name} already exist. Skipping...")
            continue


        