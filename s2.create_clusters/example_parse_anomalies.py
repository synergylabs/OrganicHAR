#!/usr/bin/env python3
"""
Simple example showing how to parse anomaly timestamps from your change detection results.
"""

import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# Example: Let's say you have a result file
def quick_parse_example():
    # Example result file path - adjust this to your actual file
    result_file = "/Volumes/Research-Prasoon/OrganicHAR/inhome_evaluation/P3-data-collection/change_detection_5.0_0.5/P3-00-20231201_140000_20231201_150000_pose_depth_top_1.json"
    
    try:
        with open(result_file, 'r') as f:
            anomaly_results = json.load(f)
        
        print("=== QUICK ANOMALY PARSING EXAMPLE ===")
        print(f"Found {len(anomaly_results)} anomaly windows\n")
        
        for i, result in enumerate(anomaly_results):
            print(f"Anomaly Window {i+1}:")
            print(f"  Time range: {result['start']:.1f}s - {result['end']:.1f}s")
            print(f"  Anomaly score: {result['score']:.3f} (lower = more anomalous)")
            
            # Extract the most anomalous moments within this window
            if 'min' in result:
                print(f"  Most anomalous moments:")
                for j, min_event in enumerate(result['min']):
                    print(f"    #{j+1}: {min_event['time']:.1f}s (frame {min_event['frame']})")
                    if 'importance' in min_event:
                        print(f"        Feature importance: {min_event['importance']}")
            
            print()
    
    except FileNotFoundError:
        print(f"File not found: {result_file}")
        print("Please update the file path to match your actual results")

def parse_all_sensors_for_session():
    """Example showing how to parse all sensors for a session"""
    
    # Your session details
    session_key = "P3-00-20231201_140000_20231201_150000"
    session_start_time = "20231201_140000"  # Extract from session_key
    base_dir = "/Volumes/Research-Prasoon/OrganicHAR/inhome_evaluation/P3-data-collection/change_detection_5.0_0.5"
    
    import os
    from datetime import datetime, timedelta
    
    session_start_dt = datetime.strptime(session_start_time, "%Y%m%d_%H%M%S")
    all_anomalies = []
    
    print(f"=== PARSING ALL SENSORS FOR SESSION {session_key} ===\n")
    
    # Find all anomaly files for this session
    for filename in os.listdir(base_dir):
        if filename.startswith(session_key) and filename.endswith('.json'):
            sensor_name = filename.replace(session_key + '_', '').replace('_top_1.json', '')
            filepath = os.path.join(base_dir, filename)
            
            print(f"Sensor: {sensor_name}")
            
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            for result in results:
                if 'min' in result:
                    for min_event in result['min']:
                        anomaly_seconds = min_event['time']
                        real_timestamp = session_start_dt + timedelta(seconds=anomaly_seconds)
                        
                        anomaly_info = {
                            'sensor': sensor_name,
                            'time_seconds': anomaly_seconds,
                            'real_timestamp': real_timestamp,
                            'score': result['score'],
                            'frame': min_event['frame']
                        }
                        all_anomalies.append(anomaly_info)
                        
                        print(f"  Anomaly at {real_timestamp.strftime('%H:%M:%S')} ({anomaly_seconds:.1f}s) - score: {result['score']:.3f}")
            print()
    
    # Sort all anomalies by time
    all_anomalies.sort(key=lambda x: x['time_seconds'])
    
    print("=== ALL ANOMALIES CHRONOLOGICALLY ===")
    for anomaly in all_anomalies:
        print(f"{anomaly['real_timestamp'].strftime('%H:%M:%S')} - {anomaly['sensor']} (score: {anomaly['score']:.3f})")

if __name__ == "__main__":
    print("Choose parsing method:")
    print("1. Quick example (single file)")
    print("2. Parse all sensors for session")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        quick_parse_example()
    elif choice == "2":
        parse_all_sensors_for_session()
    else:
        print("Invalid choice!") 