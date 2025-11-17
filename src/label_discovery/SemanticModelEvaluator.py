import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI
import json
import os
import os
import pandas as pd
import numpy as np
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import dotenv
from sentence_transformers import SentenceTransformer
from functools import partial
import ollama
import tqdm



class SemanticModelEvaluator:
    def __init__(self, labels: List[str], location: str,
                 embedding_model: str = "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M",
                 llm_model: str = "gpt-4.1",
                 cache_file: Optional[str] = None,
                 device: str = "mps"):
        """Initialize evaluator with labels and location context"""
        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.labels = labels
        self.location = location
        self.cache_file = cache_file
        self.device = device
        # Load cached data
        cached_data = self._load_cached_data()
        self.label_analysis = cached_data['analyses']
        self.label_embeddings = cached_data['embeddings']
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        if not self.embedding_model.startswith("text-embedding"):
            # self.model = SentenceTransformer(self.embedding_model, device=self.device)
            self.model = partial(ollama.embed, model=self.embedding_model)

        # Initialize and compute similarity matrix
        self.similarity_matrix = self._compute_similarity_matrix()

    def _load_cached_data(self) -> Dict:
        """Load cached analysis and embeddings if available"""
        cached_data = {'analyses': {}, 'embeddings': {}}
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                # Load analyses from JSON
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    cached_data['analyses'] = cache.get('analyses', {})
                
                # Load embeddings from npz file
                embeddings_file = self.cache_file.replace('.json', '.npz')
                if os.path.exists(embeddings_file):
                    npz_data = np.load(embeddings_file)
                    cached_data['embeddings'] = {key: npz_data[key] for key in npz_data.files}
                
                print(f"Loaded {len(cached_data['analyses'])} cached analyses and {len(cached_data['embeddings'])} embeddings")
            except Exception as e:
                print(f"Error loading cache: {e}")
        return cached_data

    def _save_cached_data(self):
        """Save current analysis and embeddings to cache file"""
        if self.cache_file:
            try:
                # Save analyses as JSON
                cache_data = {'analyses': self.label_analysis}
                with open(self.cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                
                # Save embeddings as npz
                if self.label_embeddings:
                    embeddings_file = self.cache_file.replace('.json', '.npz')
                    np.savez(embeddings_file, **self.label_embeddings)
            except Exception as e:
                print(f"Error saving cache: {e}")

    def _analyze_activity_label(self, label: str) -> Dict:
        """Use cached analysis or LLM to analyze activity label"""
        # Return cached analysis if available
        if label in self.label_analysis:
            return self.label_analysis[label]

        prompt = f"""You are an expert in human activity analysis, specializing in ergonomic and functional breakdowns of kitchen tasks. Your goal is to deconstruct a given activity label into its core functional components based on the provided location.

Analyze the following kitchen activity label: "{label}"
Location Context: This activity happens at/around the {self.location}.

Given this specific location context, break down the kitchen activity into the following components. Focus on the most likely and primary interpretations. Do not list every possibility, but the most common scenario.

Return your analysis in the following strict JSON format:

{{
    "ActionVerb": "A single, primary verb describing the core physical action (e.g., 'Washing', 'Cutting', 'Placing', 'Opening', 'Measuring', 'Stirring').",
    "PrimaryTargetObject": "The single, main noun or object that the action is directly performed ON. For 'washing dishes', this is 'dishes'. For 'opening refrigerator', this is 'refrigerator door'.",
    "InstrumentalObjects": [
        "A list of tools or secondary objects USED to perform the action. For 'washing dishes', this could be ['sponge', 'dish soap', 'faucet']. For 'cutting vegetables', this is ['knife', 'cutting board']."
    ],
    "SpatialContext": {{
        "sub_location": "A concise description of the specific sub-area within the location (e.g., 'in sink basin', 'on counter next to stove', 'in front of lower dishwasher rack').",
        "key_environmental_objects": ["A list of key fixed objects that define the immediate workspace, like 'sink faucet', 'stovetop burner', 'refrigerator shelf'."]
    }},
    "FunctionalIntent": {{
        "category": "Choose ONE category from this exact list: ['Cleanliness', 'Food Preparation', 'Cooking', 'Consumption', 'Storage', 'Appliance Operation', 'Organization', 'Idle/Other']",
        "goal_description": "A brief (under 10 words) description of the ultimate goal. e.g., 'to sanitize dishware for reuse', 'to prepare ingredients for a recipe'."
    }},
    "Transfers": {{
        "is_transfer": "A boolean (true/false) indicating if the primary action involves moving material from a source to a destination.",
        "source_object": "The object FROM which material is being moved (e.g., 'measuring cup', 'egg shell'). Null if not applicable.",
        "destination_object": "The object TO which material is being moved (e.g., 'mixing bowl', 'frying pan'). Null if not applicable."
    }},
    "TangentialActivities": {{
        "preceding_activity": "The most likely activity that immediately PRECEDES this one (e.g., 'placing dirty dishes near sink' precedes 'washing dishes').",
        "succeeding_activity": "The most likely activity that immediately SUCCEEDS this one (e.g., 'placing clean dishes in drying rack' succeeds 'washing dishes')."
    }}
}}

Ensure your response is ONLY the JSON object, with no additional text or explanations.
        """

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        # Cache the new analysis
        analysis = json.loads(response.choices[0].message.content)
        self.label_analysis[label] = analysis
        self._save_cached_data()

        return analysis

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get or compute embedding for text"""
        if text not in self.label_embeddings:
            if (self.embedding_model == "text-embedding-3-small") or (self.embedding_model == "text-embedding-3-large"):
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                self.label_embeddings[text] = np.array(response.data[0].embedding)
            else:
                embed_response = self.model(input=text)
                self.label_embeddings[text] = np.array(embed_response.embeddings).flatten()
                assert self.label_embeddings[text].shape[0]  == 4096, f"Embedding shape is {self.label_embeddings[text].shape}"
            self._save_cached_data()  # Save after computing new embedding
        return self.label_embeddings[text]
    
    def _cosine_similarity(self, emb1, emb2):
        """Helper function to compute cosine similarity."""
        if emb1 is None or emb2 is None:
            return 0.0
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def _get_embedding_for_list(self, obj_list):
        """Helper to get a single embedding for a list of objects."""
        if not obj_list or not isinstance(obj_list, list):
            return self._get_embedding("") # Return embedding for empty string as a zero vector baseline
        # Embed the sorted, concatenated string of objects
        sorted_string = " ".join(sorted(obj_list))
        return self._get_embedding(sorted_string)


    def _compute_similarity(self, analysis1, analysis2):
        """
        Computes similarity between two activity analyses using the revised,
        more robust data structure.
        """
        scores = {}
        
        # --- Weights Configuration ---
        weights = {
            "ActionVerb": 0.20,
            "PrimaryTargetObject": 0.30,
            "InstrumentalObjects": 0.10,
            "FunctionalIntent": 0.15,
            "Transfers": 0.15,
            "TangentialActivities": 0.10
        }

        # 1. Action Verb Similarity (with dynamic weighting)
        verb_weight = weights["ActionVerb"]
        generic_verbs = ['Handling', 'Manipulating', 'Interacting', 'Using', 'Accessing']
        if analysis1['ActionVerb'] in generic_verbs or analysis2['ActionVerb'] in generic_verbs:
            # Reduce weight for generic verbs and redistribute it to the more important target object
            verb_weight *= 0.5 
            weights['PrimaryTargetObject'] += weights['ActionVerb'] * 0.5

        verb_emb1 = self._get_embedding(analysis1.get('ActionVerb', ''))
        verb_emb2 = self._get_embedding(analysis2.get('ActionVerb', ''))
        scores['ActionVerb'] = verb_weight * self._cosine_similarity(verb_emb1, verb_emb2)

        # 2. Primary Target Object Similarity
        target_emb1 = self._get_embedding(analysis1.get('PrimaryTargetObject', ''))
        target_emb2 = self._get_embedding(analysis2.get('PrimaryTargetObject', ''))
        scores['PrimaryTargetObject'] = weights['PrimaryTargetObject'] * self._cosine_similarity(target_emb1, target_emb2)

        # 3. Instrumental Objects Similarity
        instr_emb1 = self._get_embedding_for_list(analysis1.get('InstrumentalObjects'))
        instr_emb2 = self._get_embedding_for_list(analysis2.get('InstrumentalObjects'))
        scores['InstrumentalObjects'] = weights['InstrumentalObjects'] * self._cosine_similarity(instr_emb1, instr_emb2)

        # 4. Functional Intent Similarity
        intent1, intent2 = analysis1.get('FunctionalIntent', {}), analysis2.get('FunctionalIntent', {})
        category_sim = 1.0 if intent1.get('category') == intent2.get('category') else 0.0
        goal_emb1 = self._get_embedding(intent1.get('goal_description', ''))
        goal_emb2 = self._get_embedding(intent2.get('goal_description', ''))
        # Give category match more weight than the free-text goal
        intent_score = 0.8 * category_sim + 0.2 * self._cosine_similarity(goal_emb1, goal_emb2)
        scores['FunctionalIntent'] = weights['FunctionalIntent'] * intent_score

        # 5. Transfers Similarity (Source/Destination)
        transfer1, transfer2 = analysis1.get('Transfers', {}), analysis2.get('Transfers', {})
        if transfer1.get('is_transfer') and transfer2.get('is_transfer'):
            src_emb1 = self._get_embedding(transfer1.get('source_object', ''))
            src_emb2 = self._get_embedding(transfer2.get('source_object', ''))
            dest_emb1 = self._get_embedding(transfer1.get('destination_object', ''))
            dest_emb2 = self._get_embedding(transfer2.get('destination_object', ''))
            
            src_sim = self._cosine_similarity(src_emb1, src_emb2)
            dest_sim = self._cosine_similarity(dest_emb1, dest_emb2)
            scores['Transfers'] = weights['Transfers'] * ((src_sim + dest_sim) / 2.0)
        else:
            # If one or both are not transfers, similarity is based on whether this fact matches.
            scores['Transfers'] = weights['Transfers'] * (1.0 if not transfer1.get('is_transfer') and not transfer2.get('is_transfer') else 0.0)

        # 6. Tangential Activities (Workflow) Similarity
        workflow1, workflow2 = analysis1.get('TangentialActivities', {}), analysis2.get('TangentialActivities', {})
        pre_emb1 = self._get_embedding(workflow1.get('preceding_activity', ''))
        pre_emb2 = self._get_embedding(workflow2.get('preceding_activity', ''))
        suc_emb1 = self._get_embedding(workflow1.get('succeeding_activity', ''))
        suc_emb2 = self._get_embedding(workflow2.get('succeeding_activity', ''))
        
        pre_sim = self._cosine_similarity(pre_emb1, pre_emb2)
        suc_sim = self._cosine_similarity(suc_emb1, suc_emb2)
        scores['TangentialActivities'] = weights['TangentialActivities'] * ((pre_sim + suc_sim) / 2.0)
        
        return sum(scores.values())

    def _compute_similarity_matrix(self) -> np.ndarray:
        """Compute all pairwise similarities between labels"""
        n_labels = len(self.labels)
        similarity_matrix = np.zeros((n_labels, n_labels))

        # First analyze all labels
        for label in tqdm.tqdm(self.labels, desc="Analyzing labels"):
            if label not in self.label_analysis:
                self.label_analysis[label] = self._analyze_activity_label(label)

        # Compute similarities
        pbar = tqdm.tqdm(total=n_labels*n_labels, desc="Computing similarities")
        for i, label1 in enumerate(self.labels):
            for j, label2 in enumerate(self.labels):
                pbar.update(1)
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = self._compute_similarity(
                        self.label_analysis[label1],
                        self.label_analysis[label2]
                    )
        pbar.close()

        return similarity_matrix

    def get_semantic_accuracy(self, y_true: List[str], y_pred: List[str]) -> float:
        """Compute semantic accuracy using precomputed similarity matrix"""
        scores = []
        for true_label, pred_label in zip(y_true, y_pred):
            i = self.label_to_idx[true_label]
            j = self.label_to_idx[pred_label]
            scores.append(self.similarity_matrix[i, j])
        return np.mean(scores)





def label_hierarchical_clustering(similarity_df, threshold=0.8):
    """
    Cluster labels based on a similarity matrix.

    Parameters:
    similarity_df (pd.DataFrame): DataFrame with labels as both index and columns, containing similarity values.
    threshold (float): Threshold for forming clusters.

    Returns:
    pd.DataFrame: DataFrame with cluster_id and list of labels in each cluster.
    """
    labels = similarity_df.index.tolist()
    similarity_matrix = similarity_df.values

    # Compute the linkage matrix
    condensed_dist_matrix = squareform(1 - similarity_matrix)
    condensed_dist_matrix = np.maximum(condensed_dist_matrix, 1e-10)

    try:
        linkage_matrix = linkage(condensed_dist_matrix, method='average')
    except Exception:
        raise ValueError("Error in linkage matrix computation")

    # Form clusters
    try:
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    except Exception:
        raise ValueError("Error in forming clusters")

    # Create a DataFrame to map original labels to clusters
    label_clusters = pd.DataFrame({'label': labels, 'cluster': clusters})

    # Group labels by cluster_id
    cluster_dict = label_clusters.groupby('cluster')['label'].apply(list).reset_index()

    return cluster_dict


def generate_cluster_labels(location_sim_matrices, label_merging_threshold):
    """Generate cluster labels for all kitchens."""

    final_cluster_label_info = []

    for location_key in location_sim_matrices:
        # Get similarity matrix for this location
        location_sim_matrix = location_sim_matrices[location_key]

        # use the label as is if there is only one label
        if location_sim_matrix.shape[0] == 1:
            cluster_labels = location_sim_matrix.index.tolist()
            cluster_names = {0: x for x in cluster_labels}
            final_cluster_label_info.append((
                location_key,
                cluster_labels[0],
                label_merging_threshold,
                0,
                cluster_names[0]
            ))
        else:
            # Perform hierarchical clustering
            cluster_dict = label_hierarchical_clustering(
                location_sim_matrix,
                threshold=label_merging_threshold
            ).to_dict()

            cluster_labels = cluster_dict['label']

            # Create cluster names by joining all labels in each cluster
            cluster_names = {x: '/'.join(sorted(cluster_labels[x])) for x in cluster_labels}

            # Store cluster information
            for cluster_idx in cluster_names:
                for raw_activity in cluster_labels[cluster_idx]:
                    final_cluster_label_info.append((
                        location_key,
                        raw_activity,
                        label_merging_threshold,
                        cluster_idx,
                        cluster_names[cluster_idx]
                    ))

        print(f"Cluster | {location_key} | Total Labels: {len(cluster_labels)}")

    # Save cluster labels to CSV
    df_cluster_labels = pd.DataFrame(
        final_cluster_label_info,
        columns=['location_cluster', 'activity_cluster', 'threshold', 'cluster_id', 'cluster_activities']
    )
    return df_cluster_labels


if __name__=="__main__":
    import dotenv
    dotenv.load_dotenv()
    KITCHEN_NAME = "K6-CMU-TCS_F2"
    # get the label directory
    label_generation_dir = f"/Users/prasoon/Research/VAX/Results/ubicomp_results/label_generation_v5/{KITCHEN_NAME}"
    activity_mappings_file = f"{label_generation_dir}/merged_activities.json"
    activity_mappings = json.load(open(activity_mappings_file, "r"))

    # loop over locations and generate cache for activity matrix
    similarity_cache_dir = f"{label_generation_dir}/similarity_cache"
    os.makedirs(similarity_cache_dir, exist_ok=True)

    for location_key in activity_mappings:
        location_specific_labels = list(activity_mappings[location_key].keys())
        evaluator = SemanticModelEvaluator(location_specific_labels, location_key, cache_file=f"{similarity_cache_dir}/{location_key}.json")
        print(f"Initialized evaluator for location: {location_key}")

