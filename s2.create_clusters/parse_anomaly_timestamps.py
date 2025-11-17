import json
import os
import pandas as pd
from datetime import datetime, timedelta
import argparse

def parse_anomaly_timestamps(results_file, session_start_time=None):
    """
    Parse anomaly detection results to extract meaningful timestamps.
    
    Args:
        results_file: Path to the JSON file with anomaly detection results
        session_start_time: Session start time as datetime object or string
    
    Returns:
        DataFrame with anomaly information including real timestamps
    """
    with open(results_file, 'r') as f:
        anomaly_results = json.load(f)
    
    anomalies = []
    
    for result in anomaly_results:
        # Extract window information
        window_start = result['start']  # seconds from start of session
        window_end = result['end']
        window_score = result['score']
        
        # Extract most anomalous point in this window
        if 'min' in result and len(result['min']) > 0:
            for min_event in result['min']:
                anomaly_time_seconds = min_event['time']  # seconds from start
                anomaly_frame = min_event['frame']
                importance = min_event.get('importance', {})
                
                anomaly_info = {
                    'window_start_seconds': window_start,
                    'window_end_seconds': window_end,
                    'window_score': window_score,
                    'anomaly_time_seconds': anomaly_time_seconds,
                    'anomaly_frame': anomaly_frame,
                    'importance': importance
                }
                
                # If we have session start time, calculate real timestamp
                if session_start_time:
                    if isinstance(session_start_time, str):
                        session_start_dt = datetime.strptime(session_start_time, "%Y%m%d_%H%M%S")
                    else:
                        session_start_dt = session_start_time
                    
                    real_timestamp = session_start_dt + timedelta(seconds=anomaly_time_seconds)
                    anomaly_info['real_timestamp'] = real_timestamp
                    anomaly_info['real_timestamp_str'] = real_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                
                anomalies.append(anomaly_info)
    
    return pd.DataFrame(anomalies)

def analyze_session_anomalies(change_detection_dir, session_key, session_start_time):
    """
    Analyze all sensor anomalies for a specific session.
    
    Args:
        change_detection_dir: Base directory with anomaly detection results
        session_key: Session identifier (e.g., "P3-00-20231201_140000_20231201_150000")
        session_start_time: Start time of the session
    """
    print(f"\n=== Analyzing Anomalies for Session: {session_key} ===")
    
    all_anomalies = {}
    
    # Find all anomaly files for this session
    for filename in os.listdir(change_detection_dir):
        if filename.startswith(session_key) and filename.endswith('.json'):
            sensor_name = filename.replace(session_key + '_', '').replace('_top_1.json', '')
            filepath = os.path.join(change_detection_dir, filename)
            
            print(f"\nProcessing {sensor_name}...")
            anomalies_df = parse_anomaly_timestamps(filepath, session_start_time)
            
            if not anomalies_df.empty:
                print(f"Found {len(anomalies_df)} anomalies:")
                for _, anomaly in anomalies_df.iterrows():
                    print(f"  - Time: {anomaly.get('real_timestamp_str', f'{anomaly.anomaly_time_seconds:.1f}s')}")
                    print(f"    Score: {anomaly.window_score:.3f}")
                    if anomaly.importance:
                        print(f"    Importance: {anomaly.importance}")
                
                all_anomalies[sensor_name] = anomalies_df
            else:
                print(f"No anomalies found for {sensor_name}")
    
    return all_anomalies

def main():
    parser = argparse.ArgumentParser(description="Parse anomaly detection timestamps")
    parser.add_argument("--results_dir", required=True, help="Directory with anomaly detection results")
    parser.add_argument("--session_key", required=True, help="Session key to analyze")
    parser.add_argument("--session_start", required=True, help="Session start time (format: YYYYMMDD_HHMMSS)")
    args = parser.parse_args()
    
    # Analyze all anomalies for the session
    all_anomalies = analyze_session_anomalies(
        args.results_dir, 
        args.session_key, 
        args.session_start
    )
    
    # Combine all anomalies and sort by time
    all_combined = []
    for sensor, df in all_anomalies.items():
        df_copy = df.copy()
        df_copy['sensor'] = sensor
        all_combined.append(df_copy)
    
    if all_combined:
        combined_df = pd.concat(all_combined, ignore_index=True)
        combined_df = combined_df.sort_values('anomaly_time_seconds')
        
        print(f"\n=== ALL ANOMALIES SORTED BY TIME ===")
        for _, anomaly in combined_df.iterrows():
            timestamp_str = anomaly.get('real_timestamp_str', f'{anomaly.anomaly_time_seconds:.1f}s')
            print(f"{timestamp_str} - {anomaly.sensor} (score: {anomaly.window_score:.3f})")
        
        # Save combined results
        output_file = f"{args.results_dir}/{args.session_key}_all_anomalies.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\nSaved combined results to: {output_file}")

if __name__ == "__main__":
    main() 