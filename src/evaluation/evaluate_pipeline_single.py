#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
import ast
import sys
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_predictions(df_gt, df_pred, output_file=None):
    """
    Evaluate activity predictions against ground truth, normalizing both to the same label space.

    Args:
        df_gt (pd.DataFrame): Ground truth dataframe
        df_pred (pd.DataFrame): Predictions dataframe
        output_file (str, optional): Path to save evaluation results

    Returns:
        dict: Evaluation metrics
    """
    # Extract snippet_id from instance_id in predictions if it doesn't exist
    if 'window_id' not in df_pred.columns:
        sys.exit("window_id not found in predictions")

    # Create a merged dataframe for evaluation
    merged_df = pd.merge(
        df_gt,
        df_pred[['session_key', 'window_id', 'prediction', 'confidence']],
        on=['session_key', 'window_id'],
        how='inner'
    )
    if merged_df.shape[0]==0:
        print("No matching rows found between ground truth and predictions.")
        return {}

    # Process each row to determine if prediction is correct and normalize labels
    results = []

    # Lists for final normalized labels used for metrics calculation
    final_y_true = []
    final_y_pred = []
    all_final_labels = set()

    for _, row in merged_df.iterrows():
        # Get the predicted activity and expand it to ground truth labels
        pred_activities = [row['prediction']]

        # Get the actual ground truth labels
        mapped_gt_labels = row['av_mapped_gt_label']

        # Check if there's an overlap between prediction and ground truth
        has_overlap = any(label in mapped_gt_labels for label in pred_activities)

        # Format the prediction as an OR-separated string
        normalized_pred = " OR ".join(sorted(pred_activities)) if pred_activities else "unknown"

        # Normalize ground truth to be the same as prediction if there's overlap
        if has_overlap:
            normalized_gt = normalized_pred
            is_correct = True
        else:
            # If no overlap, keep the original ground truth labels
            normalized_gt = mapped_gt_labels[0] if mapped_gt_labels else "unknown"
            is_correct = False

        # Add normalized labels to the sets for confusion matrix
        all_final_labels.add(normalized_pred)
        all_final_labels.add(normalized_gt)

        # For this instance, add to the final y_true and y_pred lists
        final_y_true.append(normalized_gt)
        final_y_pred.append(normalized_pred)

        results.append({
            'session_key': row['session_key'],
            'window_id': row['window_id'],
            'actual_gt': row['gt_label'],
            'pred_activity': pred_activities[0],
            # 'humanized_pred': row['humanized_prediction'],
            'mapped_gt': mapped_gt_labels,
            'normalized_gt': normalized_gt,
            'normalized_pred': normalized_pred,
            'is_correct': is_correct,
            'confidence': row['confidence'],
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate overall accuracy based on exact matches of normalized labels
    correct_count = sum(1 for gt, pred in zip(final_y_true, final_y_pred) if gt == pred)
    total_count = len(final_y_true)
    accuracy = correct_count / total_count if total_count > 0 else 0

    # Convert labels to numerical form for sklearn metrics
    all_final_labels = sorted(all_final_labels)
    label_to_idx = {label: i for i, label in enumerate(all_final_labels)}

    # Create one-hot encoded arrays for macro precision/recall
    y_true_encoded = np.zeros((len(final_y_true), len(all_final_labels)))
    y_pred_encoded = np.zeros((len(final_y_pred), len(all_final_labels)))

    for i, label in enumerate(final_y_true):
        y_true_encoded[i, label_to_idx[label]] = 1

    for i, label in enumerate(final_y_pred):
        y_pred_encoded[i, label_to_idx[label]] = 1

    # Calculate macro precision and recall
    
    macro_precision = precision_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
    macro_recall = recall_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)

    micro_precision = precision_score(y_true_encoded, y_pred_encoded, average='micro', zero_division=0)
    micro_recall = recall_score(y_true_encoded, y_pred_encoded, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='micro', zero_division=0)

    # Create confusion matrix for normalized labels
    cm = np.zeros((len(all_final_labels), len(all_final_labels)))

    # Fill confusion matrix
    for true_label, pred_label in zip(final_y_true, final_y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1

    # Compile metrics
    metrics = {
        'overall': {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'correct_count': correct_count,
            'total_count': total_count
        },
        'confusion_matrix': {
            'labels': all_final_labels,
            'matrix': cm.tolist()
        }
    }

    # Save detailed results to file if requested
    if output_file:
        # Save metrics
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Also save the detailed results dataframe
        results_df.to_csv(f"{output_file.rsplit('.', 1)[0]}_details.csv", index=False)

    return metrics


def plot_evaluation_results(metrics, output_dir=None):
    """
    Plot evaluation results - now including a proper confusion matrix.

    Args:
        metrics (dict): Evaluation metrics from evaluate_predictions
        output_dir (str, optional): Directory to save plots
    """
    # Overall metrics output
    # print("\nEvaluation Summary:")
    # print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
    # print(f"Macro Precision: {metrics['overall']['macro_precision']:.4f}")
    # print(f"Macro Recall: {metrics['overall']['macro_recall']:.4f}")
    # print(f"Correct predictions: {metrics['overall']['correct_count']} out of {metrics['overall']['total_count']}")

    if output_dir:
        # Get confusion matrix data
        cm_data = metrics['confusion_matrix']
        labels = cm_data['labels']
        cm = np.array(cm_data['matrix'])

        # Simplify labels for display by truncating if they're too long
        display_labels = []
        for label in labels:
            if len(label) > 40:
                display_labels.append(label[:37] + '...')
            else:
                display_labels.append(label)

        # 1. Prediction distribution (top 15 most frequent labels)
        plt.figure(figsize=(16, 10))

        # Get the count for each label
        label_sums = np.sum(cm, axis=0)  # Sum columns for predicted counts
        top_indices = np.argsort(label_sums)[::-1][:15]

        top_labels = [display_labels[i] for i in top_indices]

        # Create a bar chart of the most frequent predictions
        plt.bar(top_labels, label_sums[top_indices], color='steelblue')
        plt.xticks(rotation=90)
        plt.title('Most Frequent Prediction Labels')
        plt.ylabel('Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(f"{output_dir}/prediction_distribution.png", bbox_inches='tight')
        plt.close()

        # Confusion Matrix - full matrix with raw counts
        plt.figure(figsize=(20, 20))  # Increased height to accommodate label table

        # Create a figure with 2 subplots - top for confusion matrix, bottom for label table
        fig, (ax_cm, ax_table) = plt.subplots(2, 1, figsize=(20, 20),
                                               gridspec_kw={'height_ratios': [3, 1]})

        # Plot heat map with raw counts (not normalized) in the top subplot
        sns.heatmap(
            cm,  # Use the raw count matrix
            annot=True,  # Show annotations
            cmap='Blues',
            xticklabels=np.arange(len(display_labels)),  # Use indices instead of labels
            yticklabels=np.arange(len(display_labels)),
            fmt='g',  # Use integer format for annotations
            ax=ax_cm,
            annot_kws={'fontsize': 16},  # Adjust font size for annotations
        )
        ax_cm.set_title('Confusion Matrix (Raw Counts)')
        ax_cm.set_xlabel('Predicted Label Index')
        ax_cm.set_ylabel('True Label Index')

        # Create a table with full labels in the bottom subplot
        cell_text = [[f"{i}", labels[i]] for i in range(len(labels))]
        ax_table.axis('tight')
        ax_table.axis('off')
        label_table = ax_table.table(
            cellText=cell_text,
            colLabels=["Index", "Complete Label"],
            loc='center',
            cellLoc='left',
            colWidths=[0.05, 0.95]
        )
        label_table.auto_set_font_size(False)
        label_table.set_fontsize(9)
        label_table.scale(1, 1.5)  # Adjust table size for readability

        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_full.png", bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate activity predictions against ground truth.')
    parser.add_argument('--gt_file', required=False, help='Path to ground truth CSV file')
    parser.add_argument('--pred_file', required=False, help='Path to predictions CSV file')
    parser.add_argument('--mappings_file', required=False, help='Path to activity mappings JSON file')
    parser.add_argument('--output_file', help='Path to save evaluation results (JSON)')
    parser.add_argument('--output_dir', help='Directory to save plots')

    args = parser.parse_args()
    args.gt_file = "/Users/prasoon/Research/VAX/Results/ubicomp_results_may25/raw_gt_labels.csv"
    args.pred_file = "/Users/prasoon/Research/VAX/Results/ubicomp_results_may25/location_activity_training/P3/doppler_lidar_prismmotion_thermal/link_0.2/LeaveOneInstanceOut-predictions.csv"
    args.mappings_file = "activity_mapping_results.json"
    args.output_dir = "tmp/"
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = f"{args.output_dir}/tmp_evaluation_results.json"


    # Load the data
    df_gt = pd.read_csv(args.gt_file)
    df_pred = pd.read_csv(args.pred_file)
    df_pred = df_pred[df_pred.sensor == 'best']

    # turn the prediction to A/V values to evaluate A/V accuracy
    df_pred = df_pred[df_pred.av_label!="no label"]
    df_pred['prediction'] = df_pred['av_label']
    df_pred['score'] = 1


    # Load activity to ground truth mappings
    activity_location_to_gt_map = load_mappings(args.mappings_file)

    # Run evaluation
    metrics = evaluate_predictions(
        df_gt,
        df_pred,
        activity_location_to_gt_map,
        args.output_file
    )

    # Print summary
    print("\nEvaluation Summary:")
    print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['overall']['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['overall']['macro_recall']:.4f}")
    print(f"Correct predictions: {metrics['overall']['correct_count']} out of {metrics['overall']['total_count']}")

    # Generate plots if output_dir is provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_evaluation_results(metrics, args.output_dir)
        print(f"\nPlots saved to {args.output_dir}")