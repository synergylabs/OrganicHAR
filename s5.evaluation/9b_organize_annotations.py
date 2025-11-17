import os
import json
import pandas as pd
import argparse
import sys
import dotenv
dotenv.load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
sys.path.append(PROJECT_ROOT)

parser = argparse.ArgumentParser(description="Reorganize ground truth annotations into CSV format for comparison with predictions.")
parser.add_argument("--participant", default="eu2a2-data-collection", help="Participant name")
parser.add_argument("--window_size", default=5.0, help="Window size in seconds")
parser.add_argument("--sliding_window_length", default=0.5, help="Sliding window length in seconds")
parser.add_argument("--description_vlm_model", default="gpt-4.1-nano", help="VLM model used for description generation")
parser.add_argument("--clustering_vlm_model", default="gpt-4.1", help="VLM model used for location and activity clustering")
parser.add_argument("--merging_threshold", default=0.3, help="Threshold for merging similar labels")
parser.add_argument("--early_samples", default=60, type=int, help="Number of samples from early sessions")
parser.add_argument("--middle_samples", default=60, type=int, help="Number of samples from middle sessions")
parser.add_argument("--late_samples", default=130, type=int, help="Number of samples from late sessions")
parser.add_argument("--debug", action="store_true", help="Debug mode")

args = parser.parse_args()

participant = args.participant
window_size = args.window_size
sliding_window_length = args.sliding_window_length
description_vlm_model = args.description_vlm_model
clustering_vlm_model = args.clustering_vlm_model
merging_threshold = args.merging_threshold
early_samples = args.early_samples
middle_samples = args.middle_samples
late_samples = args.late_samples
debug = args.debug

# Calculate total target samples
total_target_samples = early_samples + middle_samples + late_samples

# Base directories
base_results_dir = "/Volumes/Research-Prasoon/OrganicHAR/inhome_evaluation"
gt_segments_base_dir = f"{base_results_dir}/{participant}/gt_segments"
gt_output_dir = f"{gt_segments_base_dir}/temporal_stratified_{total_target_samples}_samples"

# Input files
annotations_file = f"{gt_output_dir}/annotations.json"
metadata_file = f"{gt_output_dir}/samples_metadata.json"

# Output file
organized_annotations_file = f"{gt_output_dir}/organized_annotations.csv"

def load_annotations(annotations_file):
    """Load annotations from JSON file."""
    if not os.path.exists(annotations_file):
        print(f"Annotations file {annotations_file} does not exist.")
        return {}
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    return annotations

def load_metadata(metadata_file):
    """Load sample metadata from JSON file."""
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} does not exist.")
        return {}
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def extract_sample_id_from_filename(filename):
    """Extract sample_id from filename like 'sample--000--early--...'."""
    try:
        parts = filename.split('--')
        if len(parts) >= 2:
            return int(parts[1])
    except (ValueError, IndexError):
        pass
    return None

def organize_annotations(annotations, metadata):
    """Organize annotations into a structured format for CSV export."""
    # Create a lookup dictionary for sample metadata
    samples_dict = {}
    if 'samples' in metadata:
        for sample in metadata['samples']:
            samples_dict[sample['sample_id']] = sample
    
    organized_data = []
    
    for filename, annotation_data in annotations.items():
        # Extract sample_id from filename or annotation data
        sample_id = annotation_data.get('sample_id')
        if sample_id is None:
            sample_id = extract_sample_id_from_filename(filename)
        
        if sample_id is None:
            if debug:
                print(f"Could not extract sample_id from {filename}")
            continue
        
        # Get sample metadata
        sample_metadata = samples_dict.get(sample_id, {})
        
        # Extract activities and join them
        activities = annotation_data.get('activities', [])
        if isinstance(activities, list):
            gt_label = "; ".join(activities)
        else:
            gt_label = str(activities)
        
        # Create organized row
        row = {
            'sample_id': sample_id,
            'session_key': sample_metadata.get('predicted_session_key', ''),
            'training_session_key': sample_metadata.get('training_session_key', ''),
            'window_id': sample_metadata.get('window_id', ''),
            'gt_label': gt_label,
            'gt_confidence': 1.0,  # Assuming high confidence for human annotations
            'original_prediction': sample_metadata.get('prediction', ''),
            'original_confidence': sample_metadata.get('confidence', ''),
            'temporal_group': sample_metadata.get('temporal_group', ''),
            'start_time': sample_metadata.get('start_time', ''),
            'end_time': sample_metadata.get('end_time', ''),
            'comment': annotation_data.get('comment', ''),
            'annotation_timestamp': annotation_data.get('timestamp', ''),
            'video_filename': filename,
            'individual_activities': activities  # Keep original list for reference
        }
        
        organized_data.append(row)
    
    return organized_data

def main():
    print(f"Processing annotations for participant: {participant}")
    print(f"Input annotations file: {annotations_file}")
    print(f"Input metadata file: {metadata_file}")
    print(f"Output file: {organized_annotations_file}")
    
    # Load data
    annotations = load_annotations(annotations_file)
    metadata = load_metadata(metadata_file)
    
    if not annotations:
        print("No annotations found. Exiting...")
        return
    
    if not metadata:
        print("No metadata found. Exiting...")
        return
    
    print(f"Loaded {len(annotations)} annotations")
    
    # Organize annotations
    organized_data = organize_annotations(annotations, metadata)
    
    if not organized_data:
        print("No data to organize. Exiting...")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(organized_data)
    
    # Sort by sample_id for consistent ordering
    df = df.sort_values('sample_id').reset_index(drop=True)

    # only keep necessary columns
    df = df[['sample_id', 'session_key', 'window_id', 'gt_label', 'temporal_group']]
    
    # Save to CSV
    df.to_csv(organized_annotations_file, index=False)
    
    print(f"Organized {len(organized_data)} annotations into CSV format")
    print(f"Saved to: {organized_annotations_file}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total annotations: {len(df)}")
    
    if 'temporal_group' in df.columns:
        temporal_counts = df['temporal_group'].value_counts()
        print("Temporal group distribution:")
        for group, count in temporal_counts.items():
            print(f"  {group}: {count}")
    
    if 'gt_label' in df.columns:
        unique_labels = df['gt_label'].nunique()
        print(f"Unique ground truth labels: {unique_labels}")
        
        if debug:
            print("\nGround truth label distribution:")
            label_counts = df['gt_label'].value_counts()
            for label, count in label_counts.head(10).items():
                print(f"  '{label}': {count}")
            if len(label_counts) > 10:
                print(f"  ... and {len(label_counts) - 10} more")
    
    # Display first few rows for verification
    if debug:
        print("\n=== First 5 rows ===")
        print(df[['sample_id', 'session_key', 'window_id', 'gt_label', 'temporal_group']].head())

if __name__ == "__main__":
    main()
