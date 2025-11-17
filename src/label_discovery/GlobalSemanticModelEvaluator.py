import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from functools import partial
import ollama
import tqdm

# It's good practice to have utility functions outside the main class if they can stand alone.
def label_hierarchical_clustering(similarity_df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Performs hierarchical clustering on a given similarity matrix.

    Args:
        similarity_df (pd.DataFrame): A square DataFrame with labels as index and columns,
                                      containing similarity values between 0 and 1.
        threshold (float): The similarity threshold for forming clusters (e.g., 0.8).

    Returns:
        pd.DataFrame: A DataFrame mapping original labels to their assigned cluster ID.
    """
    if similarity_df.shape[0] < 2:
        return pd.DataFrame({'label': similarity_df.index, 'cluster': 1})

    labels = similarity_df.index.tolist()
    # The distance matrix is 1 - similarity
    distance_matrix = 1 - similarity_df.values
    distance_matrix = np.maximum(distance_matrix, 0) # Ensure non-negative
    
    condensed_dist_matrix = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist_matrix, method='average')

    # Note: The threshold for fcluster is for distance, so it's 1 - similarity_threshold
    cluster_ids = fcluster(linkage_matrix, 1 - threshold, criterion='distance')
    return pd.DataFrame({'label': labels, 'cluster': cluster_ids})

class GlobalSemanticModelEvaluator:
    """
    A global evaluator to compute semantic similarity across activities from different
    locations, using a weighted additive model for semantic and spatial scores.
    """
    def __init__(self,
                 activities: List[Tuple[str, str]],
                 embedding_model: str = "nomic-embed-text",
                 llm_model: str = "gpt-4.1",
                 cache_dir: Optional[str] = "cache",
                 device: str = "mps"):
        """
        Initializes the global evaluator.

        Args:
            activities (List[Tuple[str, str]]): List of (location_name, activity_label) tuples.
            embedding_model (str): Name of the embedding model.
            llm_model (str): Name of the LLM for analysis.
            cache_dir (Optional[str]): Directory to store separate cache files for analyses and embeddings.
            device (str): Device for the embedding model ('cpu', 'cuda', 'mps').
        """
        self.client = OpenAI()
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        self.activities = sorted(list(set(activities))) # Ensure unique, sorted activities
        self.device = device

        # --- Refactored Caching ---
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.analyses_cache_file = os.path.join(self.cache_dir, "analyses.json")
            self.embeddings_cache_file = os.path.join(self.cache_dir, "embeddings.npz")
        else:
            self.analyses_cache_file = None
            self.embeddings_cache_file = None

        self.analyses_cache = self._load_analyses_cache()
        self.embeddings_cache = self._load_embeddings_cache()

        if not self.embedding_model_name.startswith("text-embedding"):
            self.embedding_model = partial(ollama.embed, model=self.embedding_model_name)
        else:
            self.embedding_model = self.client.embeddings.create
        
        # This will be computed on initialization
        self.similarity_matrix = self._compute_similarity_matrix()

    def _load_analyses_cache(self) -> Dict:
        """Loads cached analyses from a JSON file."""
        if self.analyses_cache_file and os.path.exists(self.analyses_cache_file):
            try:
                with open(self.analyses_cache_file, 'r') as f:
                    analyses = json.load(f)
                print(f"Loaded {len(analyses)} cached analyses from {self.analyses_cache_file}")
                return analyses
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading analyses cache: {e}")
        return {}

    def _load_embeddings_cache(self) -> Dict:
        """Loads cached embeddings from an NPZ file."""
        if self.embeddings_cache_file and os.path.exists(self.embeddings_cache_file):
            try:
                with np.load(self.embeddings_cache_file) as data:
                    embeddings = dict(data)
                print(f"Loaded {len(embeddings)} cached embeddings from {self.embeddings_cache_file}")
                return embeddings
            except Exception as e:
                print(f"Error loading embeddings cache: {e}")
        return {}

    def _save_analyses_cache(self):
        """Saves the analyses cache to a JSON file."""
        if self.analyses_cache_file:
            try:
                with open(self.analyses_cache_file, 'w') as f:
                    json.dump(self.analyses_cache, f, indent=2)
            except IOError as e:
                print(f"Error saving analyses cache: {e}")

    def _save_embeddings_cache(self):
        """Saves the embeddings cache to an NPZ file."""
        if self.embeddings_cache_file:
            try:
                np.savez(self.embeddings_cache_file, **self.embeddings_cache)
            except IOError as e:
                print(f"Error saving embeddings cache: {e}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Gets an embedding for a given text, using cache if available."""
        if not text:
            return np.zeros(768) # Default dimension for nomic-embed-text

        if text in self.embeddings_cache:
            return self.embeddings_cache[text]

        try:
            if self.embedding_model_name.startswith("text-embedding"):
                response = self.embedding_model(model=self.embedding_model_name, input=[text])
                embedding = np.array(response.data[0].embedding)
            else: # ollama
                response = self.embedding_model(input=text)
                embedding = np.array(response.embeddings[0])

            self.embeddings_cache[text] = embedding
            self._save_embeddings_cache() # Save immediately after a new computation
            return embedding
        except Exception as e:
            print(f"Error getting embedding for '{text}': {e}")
            return np.zeros(768)

    def _analyze_activity_label(self, label: str, location: str) -> Dict:
        """Uses an LLM to analyze a single activity label within its specific location context."""
        cache_key = f"{location}::{label}"
        if cache_key in self.analyses_cache:
            return self.analyses_cache[cache_key]

        prompt = f"""You are an expert in human activity analysis, specializing in ergonomic and functional breakdowns of kitchen tasks. Your goal is to deconstruct a given activity label into its core functional components based on the provided location.

Analyze the following kitchen activity label: "{label}"
Location Context: This activity happens at/around the {location}.

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
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            analysis = json.loads(response.choices[0].message.content)
            self.analyses_cache[cache_key] = analysis
            self._save_analyses_cache() # Save immediately after a new computation
            return analysis
        except Exception as e:
            print(f"Error analyzing label '{label}' at location '{location}': {e}")
            return {}

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(emb1, emb2) / (norm1 * norm2)

    def _get_embedding_for_list(self, obj_list: List[str]) -> np.ndarray:
        if not obj_list: return self._get_embedding("")
        return self._get_embedding(" ".join(sorted(obj_list)))
    
    def _compute_semantic_similarity(self, analysis1: Dict, analysis2: Dict) -> float:
        """Computes the semantic similarity between two activity analyses (location-agnostic)."""
        if not analysis1 or not analysis2: return 0.0
        
        weights = {
            "ActionVerb": 0.20, "PrimaryTargetObject": 0.30, "InstrumentalObjects": 0.10,
            "FunctionalIntent": 0.15, "Transfers": 0.15, "TangentialActivities": 0.10
        }
        scores = {}
        verb_weight = weights["ActionVerb"]
        generic_verbs = ['Handling', 'Manipulating', 'Interacting', 'Using', 'Accessing']
        if analysis1.get('ActionVerb') in generic_verbs or analysis2.get('ActionVerb') in generic_verbs:
            verb_weight *= 0.5
            weights['PrimaryTargetObject'] += weights['ActionVerb'] * 0.5

        verb_emb1 = self._get_embedding(analysis1.get('ActionVerb', ''))
        verb_emb2 = self._get_embedding(analysis2.get('ActionVerb', ''))
        scores['ActionVerb'] = verb_weight * self._cosine_similarity(verb_emb1, verb_emb2)

        target_emb1 = self._get_embedding(analysis1.get('PrimaryTargetObject', ''))
        target_emb2 = self._get_embedding(analysis2.get('PrimaryTargetObject', ''))
        scores['PrimaryTargetObject'] = weights['PrimaryTargetObject'] * self._cosine_similarity(target_emb1, target_emb2)

        instr_emb1 = self._get_embedding_for_list(analysis1.get('InstrumentalObjects'))
        instr_emb2 = self._get_embedding_for_list(analysis2.get('InstrumentalObjects'))
        scores['InstrumentalObjects'] = weights['InstrumentalObjects'] * self._cosine_similarity(instr_emb1, instr_emb2)

        intent1, intent2 = analysis1.get('FunctionalIntent', {}), analysis2.get('FunctionalIntent', {})
        category_sim = 1.0 if intent1.get('category') == intent2.get('category') else 0.0
        goal_emb1 = self._get_embedding(intent1.get('goal_description', ''))
        goal_emb2 = self._get_embedding(intent2.get('goal_description', ''))
        intent_score = 0.8 * category_sim + 0.2 * self._cosine_similarity(goal_emb1, goal_emb2)
        scores['FunctionalIntent'] = weights['FunctionalIntent'] * intent_score

        transfer1, transfer2 = analysis1.get('Transfers', {}), analysis2.get('Transfers', {})
        if transfer1.get('is_transfer') and transfer2.get('is_transfer'):
            src_emb1, src_emb2 = self._get_embedding(transfer1.get('source_object', '')), self._get_embedding(transfer2.get('source_object', ''))
            dest_emb1, dest_emb2 = self._get_embedding(transfer1.get('destination_object', '')), self._get_embedding(transfer2.get('destination_object', ''))
            scores['Transfers'] = weights['Transfers'] * ((self._cosine_similarity(src_emb1, src_emb2) + self._cosine_similarity(dest_emb1, dest_emb2)) / 2.0)
        else:
            scores['Transfers'] = weights['Transfers'] * (1.0 if not transfer1.get('is_transfer') and not transfer2.get('is_transfer') else 0.0)

        workflow1, workflow2 = analysis1.get('TangentialActivities', {}), analysis2.get('TangentialActivities', {})
        pre_emb1, pre_emb2 = self._get_embedding(workflow1.get('preceding_activity', '')), self._get_embedding(workflow2.get('preceding_activity', ''))
        suc_emb1, suc_emb2 = self._get_embedding(workflow1.get('succeeding_activity', '')), self._get_embedding(workflow2.get('succeeding_activity', ''))
        scores['TangentialActivities'] = weights['TangentialActivities'] * ((self._cosine_similarity(pre_emb1, pre_emb2) + self._cosine_similarity(suc_emb1, suc_emb2)) / 2.0)
        
        return sum(scores.values())

    def _get_location_proximity_factor(self, loc1: str, loc2: str) -> float:
        """Calculates the proximity between two locations using semantic embeddings."""
        if loc1 == loc2: return 1.0
        loc1_emb = self._get_embedding(loc1.replace('_', ' '))
        loc2_emb = self._get_embedding(loc2.replace('_', ' '))
        return self._cosine_similarity(loc1_emb, loc2_emb)

    def _compute_final_similarity(self, analysis1: Dict, loc1: str, analysis2: Dict, loc2: str) -> float:
        """
        Computes the final similarity score using the robust weighted additive model.
        """
        # --- Define weights for semantic vs. spatial importance ---
        w_semantic = 0.5  # The "what" and "where" are equally important
        w_location = 0.5  # The "where" is a secondary factor

        semantic_score = self._compute_semantic_similarity(analysis1, analysis2)
        location_factor = self._get_location_proximity_factor(loc1, loc2)

        final_score = (w_semantic * semantic_score) + (w_location * location_factor)
        return final_score

    def _compute_similarity_matrix(self) -> pd.DataFrame:
        """Computes the all-pairs similarity matrix for the global list of activities."""
        n_activities = len(self.activities)
        similarity_matrix = np.zeros((n_activities, n_activities))

        print("Analyzing all activities...")
        for loc, label in tqdm.tqdm(self.activities, desc="Analyzing activities"):
            self._analyze_activity_label(label, loc)

        print("Computing global similarity matrix...")
        for i in range(n_activities):
            for j in tqdm.tqdm(range(i, n_activities), desc=f"Comparing activity {i+1}/{n_activities}", leave=False):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    continue

                loc1, label1 = self.activities[i]
                loc2, label2 = self.activities[j]

                analysis1 = self.analyses_cache.get(f"{loc1}::{label1}")
                analysis2 = self.analyses_cache.get(f"{loc2}::{label2}")

                score = self._compute_final_similarity(analysis1, loc1, analysis2, loc2)
                similarity_matrix[i, j] = score
                similarity_matrix[j, i] = score # Matrix is symmetric

        multi_index = pd.MultiIndex.from_tuples(self.activities, names=['location', 'activity'])
        return pd.DataFrame(similarity_matrix, index=multi_index, columns=multi_index)


if __name__ == '__main__':
    # --- Example Usage ---
    activity_mappings = {
        "kitchen_sink_zone": {"washing_dishes": 10, "rinsing_hands": 5},
        "counter_near_sink": {"washing_dishes_on_counter": 2, "chopping_vegetables": 8},
        "stovetop_zone": {"stirring_pot": 12, "placing_pan": 3}
    }

    global_activities_list = []
    for location, labels in activity_mappings.items():
        for label in labels.keys():
            global_activities_list.append((location, label))
    
    print(f"Loaded a total of {len(global_activities_list)} activities.")

    evaluator = GlobalSemanticModelEvaluator(
        activities=global_activities_list,
        cache_dir="global_cache" # Specify a directory for the caches
    )

    global_similarity_df = evaluator.similarity_matrix
    print("\nGlobal Similarity Matrix DataFrame:")
    print(global_similarity_df)

    merging_threshold = 0.7
    
    # Create a simple index for the clustering function
    simple_index_df = global_similarity_df.copy()
    simple_index_df.index = [f"{loc}::{lbl}" for loc, lbl in simple_index_df.index]
    simple_index_df.columns = simple_index_df.index

    clusters_df = label_hierarchical_clustering(simple_index_df, threshold=merging_threshold)
    
    clusters_df[['location', 'activity_cluster']] = clusters_df['label'].str.split('::', expand=True)
    cluster_groups = clusters_df.groupby('cluster')['label'].apply(lambda x: '/'.join(sorted(x.tolist())))
    clusters_df['merged_label'] = clusters_df['cluster'].map(cluster_groups)
    
    print(f"\nFinal Merged Labels (Threshold: {merging_threshold}):")
    print(clusters_df[['location', 'activity_cluster', 'cluster', 'merged_label']])
