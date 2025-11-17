import os
import pandas as pd
import json
import argparse
import sys
from datetime import datetime
import warnings
import numpy as np
import dotenv
dotenv.load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
sys.path.append(PROJECT_ROOT)
from src.evaluation.evaluate_pipeline_single import evaluate_predictions
warnings.filterwarnings("ignore")

# Same argument structure as 8d_run_predictions.py
parser = argparse.ArgumentParser(description="Merge ensemble predictions with ground truth using AV mappings")
parser.add_argument("--participant", default="eu2a2-data-collection", help="Participant name")
parser.add_argument("--session_prefix", default="P12", help="Session prefix")
parser.add_argument("--window_size", default=5.0, help="Window size in seconds")
parser.add_argument("--sliding_window_length", default=0.5, help="Sliding window length in seconds")
parser.add_argument("--description_vlm_model", required=True, help="VLM model used for description generation")
parser.add_argument("--clustering_vlm_model", default="gpt-4.1", help="VLM model used for clustering")
parser.add_argument("--merging_threshold", default=0.3, help="Threshold for merging similar labels")
parser.add_argument("--alpha", required=True, type=float, help="Alpha for clustering")
parser.add_argument("--prob_threshold", required=True, type=float, help="Probability threshold for predictions")
parser.add_argument("--humanizer_model", default="gpt-4.1", help="Humanizer model used for humanizing labels")
parser.add_argument("--early_samples", default=60, type=int, help="Number of samples from early sessions")
parser.add_argument("--middle_samples", default=60, type=int, help="Number of samples from middle sessions")
parser.add_argument("--late_samples", default=130, type=int, help="Number of samples from late sessions")
parser.add_argument("--direction", default="forward", choices=["forward", "backward"], help="Selection direction")
parser.add_argument("--debug", action="store_true", help="Debug mode")

args = parser.parse_args()

# Base directories
base_results_dir = os.environ.get("BASE_RESULTS_DIR", "./results")
base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")

def get_session_data():
    """Get session data similar to other pipeline scripts"""
    watch_sessions_file = f"{base_processed_dir}/{args.participant}/watch_sessions_filtered.json"
    if not os.path.exists(watch_sessions_file):
        print(f"File {watch_sessions_file} does not exist. Exiting...")
        exit(1)
    with open(watch_sessions_file, "r") as f:
        watch_sessions = json.load(f)
    
    base_video_dir = f"{base_processed_dir}/{args.participant}/processed_video_data"
    base_depth_dir = f"{base_processed_dir}/{args.participant}/processed_depth_data"

    def video_file_exists(base_filename, video_dir, depth_dir):
        if os.path.exists(os.path.join(video_dir, base_filename)):
            return True
        if os.path.exists(os.path.join(depth_dir, base_filename)):
            return True
        return False

    collected_sessions = [
        {
            "session_key": f"{args.session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}",
            **session_data,
        }
        for session_idx, session_data in enumerate(watch_sessions)
        if video_file_exists(f"{args.session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}.mp4", base_video_dir, base_depth_dir)
    ]
    return collected_sessions

def load_ground_truth_annotations():
    """Load organized ground truth annotations"""
    total_target_samples = args.early_samples + args.middle_samples + args.late_samples
    gt_segments_base_dir = f"{base_results_dir}/{args.participant}/gt_segments"
    gt_output_dir = f"{gt_segments_base_dir}/temporal_stratified_{total_target_samples}_samples"
    gt_file = f"{gt_output_dir}/organized_annotations.csv"
    
    if not os.path.exists(gt_file):
        print(f"Organized ground truth file not found: {gt_file}")
        return None
    
    df_gt = pd.read_csv(gt_file)
    return df_gt

def load_av_mappings(session_key):
    """Load AV mappings for a specific session"""
    label_generation_base_dir = f"{base_results_dir}/{args.participant}/label_generation_{args.window_size}_{args.sliding_window_length}/{args.description_vlm_model}"
    label_generation_dir = os.path.join(label_generation_base_dir, session_key)
    
    # Look for the GT mappings CSV file (from 9c script)
    mappings_file = os.path.join(label_generation_dir, f"gt_av_mappings_{args.merging_threshold}_{args.clustering_vlm_model}_alpha_{args.alpha}.json")
    
    if not os.path.exists(mappings_file):
        print(f"AV mappings file not found: {mappings_file}")
        return None
    
    # Load mappings CSV and convert to dictionary
    gt_av_mappings = json.load(open(mappings_file))
    
    # Create a mapping dictionary from ground_truth_label to list of vlm_generated_labels
    av_mappings = {key: value['mapped_labels'] for key, value in gt_av_mappings.items()}    
    return av_mappings

def process_session(session_idx, session_key, session_index_map):
    """Process a single session to merge ensemble predictions with ground truth"""
    
    # Load ensemble predictions
    activity_predictions_base_dir = f"{base_results_dir}/{args.participant}/activity_predictions_v3_alpha_{args.alpha}_{args.window_size}_{args.sliding_window_length}/{args.description_vlm_model}/{args.merging_threshold}_control_{args.clustering_vlm_model}/{args.direction}"
    session_predictions_dir = f"{activity_predictions_base_dir}/{session_key}"
    ensemble_predictions_dir = f"{session_predictions_dir}/ensemble_predictions"
    ensemble_file = f"{ensemble_predictions_dir}/predictions.csv"
    
    if not os.path.exists(ensemble_file):
        # print(f"Ensemble predictions file not found: {ensemble_file}")
        return False
    
    df_ensemble = pd.read_csv(ensemble_file)
    
    # Load ground truth annotations
    df_gt = load_ground_truth_annotations()
    if df_gt is None:
        return False
    
    if len(df_gt) == 0:
        print(f"No ground truth found for session {session_key}")
        return False
    
    # Load AV mappings
    av_mappings = load_av_mappings(session_key)
    if av_mappings is None:
        print(f"No AV mappings found for session {session_key}")
        return False
    
    # get activity clusters file
    label_generation_dir = f"{base_results_dir}/{args.participant}/label_generation_{args.window_size}_{args.sliding_window_length}/{args.description_vlm_model}/{session_key}"
    activity_clusters_file = os.path.join(label_generation_dir, f"optimal_activity_clusters_{args.alpha}.csv")
    if not os.path.exists(activity_clusters_file):
        print(f"Activity clusters file {activity_clusters_file} does not exist. Skipping...")
        return False
    df_activity_clusters = pd.read_csv(activity_clusters_file)

    # get the mapping from every cluster id to the list of activity clusters
    cluster_id_to_activity_clusters = df_activity_clusters.groupby('cluster_id')['activity'].apply(list).to_dict()
    df_activity_clusters['training_label'] = df_activity_clusters.apply(lambda row: " OR ".join(sorted(cluster_id_to_activity_clusters[row['cluster_id']])), axis=1)
    training_labels_map = {row['cluster_id']: row['training_label'] for _, row in df_activity_clusters.iterrows()}

    # change the labels in the ensemble predictions to the training labels
    df_ensemble['prediction'] = df_ensemble['prediction'].apply(lambda x: training_labels_map[int(x.split("-")[0])])

    # separate the gt labels into a list of labels
    df_gt['gt_label_list'] = df_gt['gt_label'].apply(lambda x: x.split("; ") if type(x) == str else ["No Person"])
    # print("Total GT labels: ", len(df_gt))
    # remove gt rows where the gt label is either No Person or No Video
    df_gt = df_gt[~df_gt['gt_label'].isin(['No Person', 'No Video','Unknown'])]
    # print("Total GT labels after removing No Person, No Video, and Unknown: ", len(df_gt))
    
    # Map ground truth labels to AV space (following the pattern from the attached code)
    df_gt['av_mapped_gt_label'] = df_gt['gt_label_list'].apply(lambda x: [av_mappings[label] for label in x if label in av_mappings])
    # print("Total GT labels after mapping: ", len(df_gt))
    # filter out rows where the av_mapped_gt_label is empty
    df_gt = df_gt[df_gt['av_mapped_gt_label'].notnull()]
    df_gt = df_gt[~df_gt['av_mapped_gt_label'].apply(lambda x: len(x) == 0)]
    # print("Total GT labels after filtering out empty mappings: ", len(df_gt))

    # concatenate the av_mapped_gt_label list
    df_gt['av_mapped_gt_label'] = df_gt['av_mapped_gt_label'].apply(lambda x: [item for sublist in x for item in sublist])
    df_gt = df_gt[['session_key', 'window_id', 'gt_label', 'temporal_group', 'av_mapped_gt_label']]

    # only keep the prediction windows that are in gt
    df_pred = df_ensemble[df_ensemble['window_id'].isin(df_gt['window_id'])]
    df_pred = df_pred[['session_key', 'window_id', 'prediction', 'confidence']]
    df_pred = df_pred[df_pred['confidence'] >= args.prob_threshold]

    # only leave session key where the session index is greater than or equal to the session index of the session key
    df_gt = df_gt[df_gt['session_key'].apply(lambda x: session_index_map[x] > session_idx)]
    df_pred = df_pred[df_pred['session_key'].apply(lambda x: session_index_map[x] > session_idx)]
    if len(df_gt) <= 20 or len(df_pred) <= 20:
        # print(f"No predictions or GT labels found for session {session_key}")
        return False
    # print("Total GT labels for predictions: ", len(df_gt))
    # print("Total predictions: ", len(df_pred))

    # evaluate the predictions
    metrics = evaluate_predictions(df_gt, df_pred, output_file=f"{ensemble_predictions_dir}/ensemble_with_gt.csv")
    if metrics == {}:
        print(f"No metrics found for session {session_key}")
        return False
    # Print summary
    acc = metrics['overall']['accuracy'] * 100
    macro_f1 = metrics['overall']['macro_f1'] * 100
    macro_precision = metrics['overall']['macro_precision'] * 100
    macro_recall = metrics['overall']['macro_recall'] * 100
    micro_f1 = metrics['overall']['micro_f1'] * 100
    micro_precision = metrics['overall']['micro_precision'] * 100
    micro_recall = metrics['overall']['micro_recall'] * 100
    correct_count = metrics['overall']['correct_count']
    total_count = metrics['overall']['total_count']
    unique_label_count = len(metrics['confusion_matrix']['labels']) 
    unique_labels = metrics['confusion_matrix']['labels']
    label_to_training_label = {label: training_labels_map.get(label, label) for label in unique_labels}
    
    # print(
    #     f"{args.participant} | {args.description_vlm_model} | {args.merging_threshold} | {args.direction} | {session_key} | {unique_label_count} | {correct_count}/{total_count} | (Acc: {int(acc)}%, F1: {int(micro_f1)}%)")
    print(
        f"{args.session_prefix[:2]} | Session index: {session_idx+1} | Unique Labels: {unique_label_count} | Correct/Total: {correct_count}/{total_count} | Accuracy: {int(acc)}%")
    # print the label_to_training_label
    # print("Discovered labels: ", end="")
    # for label, training_label in label_to_training_label.items():
    #     print(f"{training_label}", end=", ")
    return metrics

def main():
    """Main function"""
    # print(f"Starting ensemble+GT processing for participant: {args.participant}")
    # print(f"Session prefix: {args.session_prefix}")
    # print(f"Window parameters: {args.window_size}s window, {args.sliding_window_length}s step")
    # print(f"Description VLM Model: {args.description_vlm_model}")
    # print(f"Clustering VLM Model: {args.clustering_vlm_model}")
    # print(f"Merging threshold: {args.merging_threshold}")
    # print(f"Direction: {args.direction}")


    

    # Get session data
    collected_sessions = get_session_data()
    
    if not collected_sessions:
        print("No sessions found. Exiting...")
        return 1
    
    # print(f"Found {len(collected_sessions)} sessions to process")
    
    processed_count = 0
    session_index_map = {}
    for session_idx, session_data in enumerate(collected_sessions):
        session_index_map[session_data["session_key"]] = session_idx

    for session_idx, session_data in enumerate(collected_sessions):
        session_key = session_data["session_key"]
        # print(f"\nProcessing session: {session_key}")
        # print("\n\n")
        
        metrics = process_session(session_idx,session_key,session_index_map)
        if metrics is not None:
            processed_count += 1
            # save metrics to a csv file
            session_metrics_file = f"{base_results_dir}/{args.participant}/{session_key}_{args.alpha}_{args.prob_threshold}_final_metrics.json"
            json.dump(metrics, open(session_metrics_file, 'w'))
    
    # print(f"\nCompleted processing {processed_count} out of {len(collected_sessions)} sessions")
    return 0

if __name__ == "__main__":
    participants = ["eu2a2-data-collection"]
    session_prefixes = ["P12"]
    for participant, session_prefix in zip(participants, session_prefixes):
        args.participant = participant
        args.session_prefix = session_prefix
        print("\n")
        main()

