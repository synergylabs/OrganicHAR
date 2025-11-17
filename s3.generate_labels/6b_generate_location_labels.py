import os
import glob
import json
import argparse
import sys
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import dotenv

dotenv.load_dotenv()
# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.common.video_processing.KitchenLocationClustering import KitchenLocationClustering

# Base directories - adjust these paths as needed
base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")
base_results_dir = "/Volumes/Research-Prasoon/OrganicHAR/inhome_evaluation"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate location labels from video segment descriptions")
    
    # Required arguments
    parser.add_argument("--participant", default="P2-data-collection", help="Participant name (e.g., P5data-collection)")
    parser.add_argument("--session_prefix", default="P2", help="Session prefix")
    
    # Optional arguments
    parser.add_argument("--window_size", default="5.0", help="Window size in seconds (default: 5.0)")
    parser.add_argument("--sliding_window_length", default="0.5", help="Sliding window length (default: 0.5)")
    parser.add_argument("--camera_type", choices=["birdseye", "depth"], default="birdseye",
                       help="Camera type to process (default: birdseye)")
    parser.add_argument("--description_vlm_model", required=True, help="VLM model to use for description generation (default: gpt-4.1)")
    parser.add_argument("--clustering_vlm_model", default="gpt-4.1", help="VLM model to use for location generation (default: gpt-4.1)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    return parser.parse_args()

def get_session_data(base_processed_dir, participant, session_prefix, window_size, sliding_window_length):
    """Get session data similar to 5a_cluster_data.py"""
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
    return collected_sessions

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



def load_description_results(args, session_key, session_data):
    """Load description results for the given session"""
    # Setup directories based on 6a_generate_descriptions.py structure
    cache_base_dir = f"{base_results_dir}/{args.participant}/descriptions_{args.window_size}_{args.sliding_window_length}/{args.description_vlm_model}"
    cache_dir = os.path.join(cache_base_dir, args.camera_type)
    
    # Results filename from 6a_generate_descriptions.py
    results_filename = f"results_{args.participant}_{args.camera_type}_{args.window_size}_{args.sliding_window_length}_{args.description_vlm_model}.json"
    results_file = os.path.join(cache_dir, results_filename)
    
    if not os.path.exists(results_file):
        print(f"Description results not found: {results_file}")
        print("Please run 6a_generate_descriptions.py first to generate descriptions")
        return None
    
    print(f"Loading description results from: {results_file}")
    with open(results_file, 'r') as f:
        file_results = json.load(f)
        # separate the results into batch results
        batch_results = {}
        for key, value in file_results.items():
            batch_id = int(key.split('_')[-2])
            if batch_id not in batch_results:
                batch_results[batch_id] = {}
            new_key = int(key.split('_')[-1])
            if new_key in batch_results[batch_id]:
                print(f"Duplicate key found: {key}")
                print(f"Value: {value}")
                exit(1)
            batch_results[batch_id][new_key] = value
        file_results = batch_results


    all_results = {}
    # if gemini-2.5-flash, then we need to convert the results to the format of gpt-4.1
    if args.description_vlm_model == "gemini-2.5-flash":
        # load the segments batch files into an array 
        segement_batch_files = sorted(glob.glob(f"{cache_dir}/segments_file_batch_*.txt"))
        for segement_batch_file in segement_batch_files:
            with open(segement_batch_file, 'r') as f:
                segments = f.readlines()
            segments = [segment.strip() for segment in segments]
            batch_id = int(segement_batch_file.split('/')[-1].split('_')[-1].split('.')[0])
            for segment_idx, segment_file in enumerate(segments):
                segment_key = 'activity_' + segment_file.split('/')[-1].split('.mp4')[0] + '_1.0'
                if segment_idx in file_results[batch_id]:
                    all_results[segment_key] = file_results[batch_id][segment_idx]
    else:
        all_results = file_results
    
    # Parse session end time
    session_end_time = datetime.strptime(session_data['end'], "%Y%m%d_%H%M%S")
    
    # Filter results based on activity timestamps
    session_results = {}
    for activity_key, activity_data in all_results.items():
        if not activity_key.startswith('activity_'):
            continue
            
        # Parse timestamps from activity key: activity_<start>_<end>_1.0
        try:
            parts = activity_key.split('_')
            if len(parts) >= 3:
                segment_start_str = parts[1]  # 20250404144824.000000
                segment_end_str = parts[2]    # 20250404144829.000000

                # convert microseconds to milliseconds so that we can convert to datetime
                segment_start_str, segment_start_microseconds = segment_start_str.split('.')
                segment_end_str, segment_end_microseconds = segment_end_str.split('.')
                
                segment_start_str = segment_start_str + '.' + str(int(segment_start_microseconds) // 1000).zfill(3)
                segment_end_str = segment_end_str + '.' + str(int(segment_end_microseconds) // 1000).zfill(3)
                
                # Convert to datetime (remove microseconds for parsing)
                segment_start = datetime.strptime(segment_start_str, "%Y%m%d%H%M%S.%f")
                segment_end = datetime.strptime(segment_end_str, "%Y%m%d%H%M%S.%f")
                
                # Include if segment end time is within this session
                if segment_end <= session_end_time:
                    segment_end_timestamp = segment_end.timestamp()
                    session_results[segment_end_timestamp] = activity_data
                    
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse timestamp from activity key: {activity_key} - {e}")
            continue
    
    print(f"Found {len(session_results)} description results for session {session_key}")
    return session_results

def extract_locations_from_descriptions(description_results):
    """Extract location information from description results"""
    locations = []
    
    for window_id, activity_data in description_results.items():
        location_analysis = activity_data.get('location_analysis', [])
        
        for analysis in location_analysis:
            primary_location = analysis.get('primary_location', analysis.get('location', analysis.get('location_name', '')))        
            if primary_location:
                # Add to locations list with metadata
                locations.append({
                    'window_id': window_id,
                    'location': primary_location,
                    **analysis
                })
            else:
                # print(f"No location found for window {window_id}, activity_data: {activity_data}")
                ...
    
    return locations

def cluster_locations(locations_data, label_generation_dir, clustering_vlm_model):
    """Cluster locations using KitchenLocationClustering"""
    # Extract all primary locations
    all_locations = [loc_data['location'] for loc_data in locations_data]
    
    # Create location counts similar to the example
    location_counts = pd.Series(all_locations).value_counts().reset_index()
    location_counts.columns = ["location", "count"]
    kitchen_locations = location_counts.to_dict(orient="records")
    
    print(f"Found {len(kitchen_locations)} unique locations")
    
    if len(kitchen_locations) == 0:
        print("No locations found to cluster")
        return None, None, None
    
    # Initialize location clusterer
    location_clusterer = KitchenLocationClustering(model_name=clustering_vlm_model)
    clustered_locations = location_clusterer.cluster_locations(kitchen_locations)

    merged_clusters = [xr["name"] for xr in clustered_locations]
    print(f"Found {len(merged_clusters)} merged clusters")
    print(f"Merged clusters: {merged_clusters}")
    
    # Create the 1x1 map with all the locations and clusters
    location_cluster_map = {}
    for merged_location in clustered_locations:
        for location in merged_location["original_locations"]:
            location_cluster_map[location] = merged_location["name"]

    new_locations = []
    for location_data in kitchen_locations:
        if location_data["location"] not in location_cluster_map:
            new_locations.append(location_data["location"])

    max_batch_size = 100
    while len(new_locations) > 0:
        print(f"Total new locations found: {len(new_locations)}")
        new_location_map = location_clusterer.match_multiple_locations(new_locations[:max_batch_size], merged_clusters)
        found_new_locations = []
        for location in new_locations[:]:  # Create a copy to iterate over
            if location in new_location_map:
                location_cluster_map[location] = new_location_map[location]
                found_new_locations.append(location)
                new_locations.remove(location)
        if len(found_new_locations) == 0:
            for location in new_locations:
                location_cluster_map[location] = "no match"
            break

    return kitchen_locations, clustered_locations, location_cluster_map

def save_location_results(args, session_key, session_data, kitchen_locations, clustered_locations, location_cluster_map, label_generation_dir, clustering_vlm_model):
    """Save location clustering results"""
    
    # Save raw locations
    raw_locations_file = os.path.join(label_generation_dir, f"raw_locations_{clustering_vlm_model}_{session_key}.json")
    with open(raw_locations_file, "w") as f:
        json.dump(kitchen_locations, f, indent=2)
    
    # Save merged locations
    merged_locations_file = os.path.join(label_generation_dir, f"merged_locations_{clustering_vlm_model}_{session_key}.json")
    with open(merged_locations_file, "w") as f:
        json.dump(clustered_locations, f, indent=2)

    # Save location map
    location_map_file = os.path.join(label_generation_dir, f"location_map_{clustering_vlm_model}_{session_key}.json")
    with open(location_map_file, "w") as f:
        json.dump(location_cluster_map, f, indent=2)
    
    # Save summary
    summary = {
        'participant': args.participant,
        'session_prefix': args.session_prefix,
        'session_key': session_key,
        'session_start': session_data['start'],
        'session_end': session_data['end'],
        'camera_type': args.camera_type,
        'window_size': args.window_size,
        'sliding_window_length': args.sliding_window_length,
        'description_vlm_model': args.description_vlm_model,
        'clustering_vlm_model': clustering_vlm_model,
        'processing_time': datetime.now().isoformat(),
        'total_raw_locations': len(kitchen_locations),
        'total_merged_clusters': len(clustered_locations),
        'files': {
            'raw_locations': raw_locations_file,
            'merged_locations': merged_locations_file,
            'location_map': location_map_file
        }
    }
    
    summary_file = os.path.join(label_generation_dir, f"location_clustering_summary_{session_key}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Location clustering results saved:")
    print(f"  Raw locations: {raw_locations_file}")
    print(f"  Merged locations: {merged_locations_file}")
    print(f"  Location map: {location_map_file}")
    print(f"  Summary: {summary_file}")

def main():
    """Main function"""
    args = parse_args()
    
    print(f"Starting location label generation for participant: {args.participant}")
    print(f"Session prefix: {args.session_prefix}")
    print(f"Camera type: {args.camera_type}")
    print(f"Window parameters: {args.window_size}s window, {args.sliding_window_length}s step")
    print(f"Description VLM Model: {args.description_vlm_model}")
    print(f"Clustering VLM Model: {args.clustering_vlm_model}")
    
    # Get session data
    collected_sessions = get_session_data(base_processed_dir, args.participant, args.session_prefix, args.window_size, args.sliding_window_length)
    collected_sessions = collected_sessions[:12]
    
    if not collected_sessions:
        print("No sessions found. Exiting...")
        return 1
    
    # Create label generation directory
    label_generation_base_dir = f"{base_results_dir}/{args.participant}/label_generation_{args.window_size}_{args.sliding_window_length}/{args.description_vlm_model}"
    os.makedirs(label_generation_base_dir, exist_ok=True)
    
    # Loop over individual sessions
    for session_index, session_data in enumerate(collected_sessions):
        session_key = session_data["session_key"]
        
        print(f"\n=== Processing session {session_index + 1}: {session_key} ===")
        
        # Create session-specific label generation directory
        label_generation_dir = os.path.join(label_generation_base_dir, session_key)
        os.makedirs(label_generation_dir, exist_ok=True)
        
        # Check if results already exist
        locations_file = os.path.join(label_generation_dir, f"locations_{args.clustering_vlm_model}.csv")
        if os.path.exists(locations_file) and not args.debug:
            print(f"Location clustering results for {session_key} already exist. Skipping...")
            continue
        
        # Load description results for current session
        description_results = load_description_results(args, session_key, session_data)
        if description_results is None:
            print(f"No description results found for session {session_key}. Skipping...")
            continue
        
        # Extract locations from descriptions
        locations_data = extract_locations_from_descriptions(description_results)
        print(f"Extracted location data from {len(locations_data)} location analyses")
        
        # Cluster locations
        kitchen_locations, clustered_locations, location_cluster_map = cluster_locations(locations_data, label_generation_dir, args.clustering_vlm_model)
        
        if kitchen_locations is None:
            print(f"No locations to cluster for session {session_key}. Skipping...")
            continue

        df_locations = pd.DataFrame(locations_data)
        df_locations['location_cluster'] = df_locations['location'].map(location_cluster_map)
        df_locations['location_cluster'] = df_locations['location_cluster'].fillna('no_match')
        df_locations['location_cluster'] = df_locations['location_cluster'].astype(str)
        df_locations['location_cluster'] = df_locations['location_cluster'].apply(lambda x: x.replace('no_match', 'no_match'))
        df_locations['location_cluster'] = df_locations['location_cluster'].apply(lambda x: x.replace(' ', '_'))
        df_locations['location_cluster'] = df_locations['location_cluster'].apply(lambda x: x.replace('-', '_'))
        df_locations['location_cluster'] = df_locations['location_cluster'].apply(lambda x: x.replace('/', '_'))
        
        # Save the final locations file as csv
        
        df_locations.to_csv(locations_file, index=False)
        
        save_location_results(args, session_key, session_data, kitchen_locations, clustered_locations, location_cluster_map, label_generation_dir, args.clustering_vlm_model)
        
        print(f"✓ Completed location clustering for session {session_key}")
    
    print("\nLocation label generation completed!")
    return 0

if __name__ == "__main__":
    exit(main())
