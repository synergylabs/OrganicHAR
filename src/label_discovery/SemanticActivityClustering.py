import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import entropy
from typing import Tuple, Dict, List
import itertools
import warnings

# Suppress pandas warnings about fragmented DataFrames
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def _calculate_cohesion_score(
    clusters: np.ndarray,
    similarity_df: pd.DataFrame,
    counts_map: Dict[Tuple[str, str], int]
) -> float:
    """
    Calculates the count-weighted average intra-cluster similarity.

    Args:
        clusters (np.ndarray): Array of cluster assignments for each activity.
        similarity_df (pd.DataFrame): The original similarity matrix.
        counts_map (Dict): A dictionary mapping (location, activity) to its count.

    Returns:
        float: The final cohesion score.
    """
    unique_clusters = np.unique(clusters)
    total_weighted_cohesion = 0
    total_counts = 0

    for cluster_id in unique_clusters:
        member_indices = np.where(clusters == cluster_id)[0]
        
        # Skip clusters with only one member as their cohesion is undefined or trivially 1
        if len(member_indices) < 2:
            continue

        member_labels = [similarity_df.index[i] for i in member_indices]
        
        # Get all pairs of members within the cluster
        pairs = list(itertools.combinations(member_labels, 2))
        if not pairs:
            continue
            
        # Calculate average similarity for the cluster
        cluster_similarities = [similarity_df.loc[p[0], p[1]] for p in pairs]
        avg_cluster_cohesion = np.mean(cluster_similarities)
        
        # Calculate the total count for the cluster
        cluster_total_count = np.nansum([int(counts_map.get(label, 0)) for label in member_labels])

        total_weighted_cohesion += avg_cluster_cohesion * cluster_total_count
        total_counts += cluster_total_count

    if total_counts == 0:
        return 0.0

    return total_weighted_cohesion / total_counts

def _calculate_balance_score(clusters: np.ndarray, counts_map: Dict[Tuple[str, str], int], labels: pd.MultiIndex) -> float:
    """
    Calculates the balance of total counts across clusters using Shannon Entropy.

    Args:
        clusters (np.ndarray): Array of cluster assignments for each activity.
        counts_map (Dict): A dictionary mapping (location, activity) to its count.
        labels (pd.MultiIndex): The multi-index from the similarity matrix.

    Returns:
        float: The entropy-based balance score.
    """
    unique_clusters = np.unique(clusters)
    cluster_counts = []

    for cluster_id in unique_clusters:
        member_indices = np.where(clusters == cluster_id)[0]
        member_labels = [labels[i] for i in member_indices]
        cluster_total_count = np.nansum([int(counts_map.get(label, 0)) for label in member_labels])
        cluster_counts.append(cluster_total_count)

    if sum(cluster_counts) == 0:
        return 0.0
    
    # Normalize counts to create a probability distribution
    probabilities = np.array(cluster_counts) / sum(cluster_counts)
    
    # Calculate Shannon Entropy
    return entropy(probabilities, base=2)


def find_optimal_activity_clusters(
    similarity_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    alpha: float = 0.5,
    max_clusters: int = None
) -> pd.DataFrame:
    """
    Finds the optimal clustering of kitchen activities based on semantic similarity and count balance.

    Args:
        similarity_df (pd.DataFrame): A square DataFrame with a (location, activity) MultiIndex,
                                      where values represent semantic similarity.
        counts_df (pd.DataFrame): A DataFrame with columns ['location_cluster', 'activity_cluster', 'count'].
        alpha (float): The weight for the cohesion score in the final quality score calculation.
                       (1 - alpha) will be the weight for the balance score. Defaults to 0.6.
        max_clusters (int): The maximum number of clusters to test. If None, it defaults to
                            half the number of activities.

    Returns:
        pd.DataFrame: A DataFrame with the original (location, activity), its assigned cluster ID,
                      and the final merged label for that cluster.
    """
    if not isinstance(similarity_df.index, pd.MultiIndex):
        raise ValueError("similarity_df must have a MultiIndex of (location, activity).")

    # --- 1. Data Preparation ---
    
    # Create a mapping from (location, activity) to count for quick lookups
    counts_map = counts_df.set_index(['location_cluster', 'activity_cluster'])['count'].to_dict()
    
    # Ensure all activities in the similarity matrix are in the counts map, default to 0 if not found
    for activity_tuple in similarity_df.index:
        if activity_tuple not in counts_map:
            counts_map[activity_tuple] = 0

    # Convert similarity to distance for clustering
    distance_df = 1 - similarity_df
    # Scipy's linkage function requires a condensed distance matrix (a flat array)
    condensed_distance = squareform(distance_df.values)
    condensed_distance[condensed_distance <= 0] = 0

    # --- 2. Hierarchical Clustering ---
    # Using the 'ward' method, which tends to create well-sized clusters
    Z = linkage(condensed_distance, method='ward')


    # --- 3. Find Optimal Number of Clusters (k) ---
    n_activities = len(similarity_df.index)
    if max_clusters is None:
        max_clusters = n_activities // 2
    
    k_range = range(2, min(max_clusters, n_activities))
    
    if not k_range:
        print("Warning: Not enough activities to form more than one cluster. Returning single cluster.")
        # Handle the edge case of very few activities
        results_df = similarity_df.index.to_frame(index=False)
        results_df['cluster_id'] = 1
        results_df['merged_label'] = "merged_activity_1"
        return results_df


    quality_scores = []
    cohesion_scores = []
    balance_scores = []

    print(f"Evaluating optimal number of clusters from k=2 to k={max(k_range)}...")
    for k in k_range:

        clusters = fcluster(Z, t=k, criterion='maxclust')
        
        # Calculate cohesion and balance
        cohesion = _calculate_cohesion_score(clusters, similarity_df, counts_map)
        balance = _calculate_balance_score(clusters, counts_map, similarity_df.index)
        
        cohesion_scores.append(cohesion)
        balance_scores.append(balance)

    # Normalize scores to be on a 0-1 scale for fair combination
    norm_cohesion = (cohesion_scores - np.min(cohesion_scores)) / (np.max(cohesion_scores) - np.min(cohesion_scores))
    norm_balance = (balance_scores - np.min(balance_scores)) / (np.max(balance_scores) - np.min(balance_scores))
    
    # Calculate final quality score
    final_scores = alpha * norm_cohesion + (1 - alpha) * norm_balance
    
    # Find the k that maximizes the quality score
    optimal_k = k_range[np.argmax(final_scores)]
    print(f"Optimal number of clusters found: k = {optimal_k}")

    # --- 4. Generate Final Labels ---
    final_clusters = fcluster(Z, t=optimal_k, criterion='maxclust')
    
    results_df = similarity_df.index.to_frame(index=False)
    results_df.columns = ['location', 'activity']
    results_df['cluster_id'] = final_clusters
    results_df['count'] = results_df.apply(lambda row: counts_map.get((row['location'], row['activity']), 0), axis=1)

    # Determine the name for each merged cluster
    # The name will be the (location, activity) of the most frequent member of the cluster
    merged_labels_map = {}
    for cluster_id in results_df['cluster_id'].unique():
        cluster_members = results_df[results_df['cluster_id'] == cluster_id]
        # Find the member with the highest count to name the cluster
        name_giver = cluster_members.loc[cluster_members['count'].idxmax()]
        # Create a clean, descriptive merged label
        merged_labels_map[cluster_id] = f"{name_giver['location']}::{name_giver['activity']}"

    results_df['merged_label'] = results_df['cluster_id'].map(merged_labels_map)
    
    return results_df.drop(columns=['count'])


# --- Example Usage ---
if __name__ == '__main__':
    # 1. Create dummy data that mimics the real-world scenario
    
    # Sample activities and their counts
    counts_data = {
        'location_cluster': [
            'kitchen_sink_zone', 'kitchen_sink_zone', 'kitchen_sink_zone',
            'stovetop_zone', 'stovetop_zone',
            'coffee_station', 'coffee_station'
        ],
        'activity_cluster': [
            'washing_dishes', 'rinsing_plates', 'scrubbing_pot',
            'stirring_pasta', 'frying_egg',
            'operating_coffee_machine', 'adding_milk_to_coffee'
        ],
        'count': [100, 50, 30, 80, 60, 200, 40]
    }
    counts_df = pd.DataFrame(counts_data)

    # Create a MultiIndex for the similarity matrix
    activity_tuples = list(zip(counts_df['location_cluster'], counts_df['activity_cluster']))
    multi_index = pd.MultiIndex.from_tuples(activity_tuples, names=['location', 'activity'])

    # Create a dummy similarity matrix
    # In a real scenario, this would be the output of your semantic model
    similarity_data = np.array([
        [1.00, 0.95, 0.90, 0.20, 0.15, 0.05, 0.05], # washing_dishes
        [0.95, 1.00, 0.88, 0.22, 0.18, 0.06, 0.04], # rinsing_plates
        [0.90, 0.88, 1.00, 0.18, 0.12, 0.03, 0.02], # scrubbing_pot
        [0.20, 0.22, 0.18, 1.00, 0.85, 0.30, 0.25], # stirring_pasta
        [0.15, 0.18, 0.12, 0.85, 1.00, 0.25, 0.20], # frying_egg
        [0.05, 0.06, 0.03, 0.30, 0.25, 1.00, 0.80], # operating_coffee_machine
        [0.05, 0.04, 0.02, 0.25, 0.20, 0.80, 1.00], # adding_milk_to_coffee
    ])
    similarity_df = pd.DataFrame(similarity_data, index=multi_index, columns=multi_index)

    print("--- Input Similarity Matrix ---")
    print(similarity_df)
    print("\n--- Input Counts DataFrame ---")
    print(counts_df)

    # 2. Run the function to find the optimal clusters
    print("\n--- Running Clustering Algorithm ---")
    final_labels_df = find_optimal_activity_clusters(
        similarity_df=similarity_df,
        counts_df=counts_df,
        alpha=0.5, # Giving slightly more weight to semantic cohesion
        max_clusters=5
    )

    # 3. Display the results
    print("\n--- Final Merged Activity Labels ---")
    print(final_labels_df)

    # Display the groups for clarity
    print("\n--- Cluster Groups ---")
    for cluster_name, group in final_labels_df.groupby('merged_label'):
        print(f"\nCluster: {cluster_name}")
        for _, row in group.iterrows():
            print(f"  - ({row['location']}, {row['activity']})")

