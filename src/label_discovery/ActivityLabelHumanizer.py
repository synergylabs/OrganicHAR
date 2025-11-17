from typing import Dict, List, Tuple
import json
from openai import OpenAI
import os
import pandas as pd
class ActivityLabelHumanizer:
    def __init__(self, model_name: str = "gpt-4.1"):
        """
        Initialize with model name for humanizing activity labels.
        
        Args:
            model_name: Name of the language model to use
        """
        self.client = OpenAI()
        self.model_name = model_name
        self.system_prompt = """You are converting automatically generated activity cluster labels into natural, human-readable descriptions that incorporate location context meaningfully.

UNDERSTANDING THE INPUT:
These labels are machine-generated clusters of similar activities. Each label contains multiple activities joined by "OR" that were algorithmically determined to be similar. Your job is to extract the core common activity and present it in natural language.

CORE APPROACH:
1. **Extract Common Theme**: For OR-separated activities, identify the main shared activity type
2. **Incorporate Location Meaningfully**: Use location to provide helpful context, not just as a prefix
3. **Natural Language**: Write how a person would naturally describe the activity
4. **Maintain Distinctions**: Keep meaningful differences between activities, especially when location matters

LOCATION INTEGRATION PRINCIPLES:
- Use location when it adds meaningful context: "Sink prep work" vs "Counter prep work" vs "Stovetop cooking"
- Combine location + activity naturally: "Dishwasher management", "Island food prep", "Sink cleaning"
- Avoid redundant location info: Don't say "Sink sink washing" 
- Location helps distinguish similar activities in different areas

HANDLING OR-CONCATENATED LABELS:
1. Read the full OR chain to understand all included activities
2. Find the dominant common activity thread
3. Check if sub-activities need distinction or can be grouped
4. Create one clear label that covers the cluster's purpose

EXAMPLES:
- "handling bowls OR arranging items OR opening cabinets near sink" → "Sink area organization"
- "cracking eggs OR cutting ingredients OR mixing in bowl near stovetop" → "Stovetop food prep"
- "loading dishwasher OR unloading dishwasher OR reaching into dishwasher" → "Dishwasher management"
- "washing dishes OR rinsing utensils OR cleaning surfaces near sink" → "Sink cleaning tasks"

GUIDELINES:
- Use 2-4 words typically, up to 6 if needed for clarity
- Prioritize what users would want to track in their daily activities
- Keep location context when it distinguishes similar activities
- Group very similar sub-activities under one meaningful label
- **Add Key Specificity**: Include dominant tools, appliances, or dishes when they appear frequently in the OR chain
- Think: "How would someone describe this to track their kitchen routine?"

SPECIFICITY EXAMPLES:
- Tools/Equipment: "pan cooking", "bowl mixing", "knife prep", "spoon stirring"
- Appliances: "dishwasher loading", "microwave heating", "coffee machine operation"
- Food Items: "egg preparation", "bread making", "salad assembly"
- If multiple specifics: Choose the most meaningful one or use general term
- If no clear dominant element: Use broader activity description

Return the complete mapping in this JSON format:
{
    "label_mappings": {
        "original_long_activity_label_1": "human_readable_label_1",
        "original_long_activity_label_2": "human_readable_label_2",
        ...
    }
}

Include ALL activity labels shown above in your response."""

    def humanize_labels(self, location_label_pairs: List[Tuple[str, str]]) -> Dict[str, str]:
        """
        Convert ALL long labels into unique, human-interpretable ones with verification and retry logic.
        
        Args:
            location_label_pairs: List of (location, long_label) tuples
            
        Returns:
            Dictionary mapping original long labels to human-interpretable labels
        """
        if not location_label_pairs:
            return {}
        
        # Get all unique labels that need mapping
        all_labels = set(label for _, label in location_label_pairs)
        final_mappings = {}
        remaining_pairs = location_label_pairs.copy()
        max_attempts = 10
        
        for attempt in range(max_attempts):
            if not remaining_pairs:
                break
                
            prompt = self._create_prompt(remaining_pairs)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                )
                
                results = json.loads(response.choices[0].message.content)
                current_mappings = results.get("label_mappings", {})
                
                # Add successful mappings to final results
                final_mappings.update(current_mappings)
                
                # Find which labels still need mapping
                mapped_labels = set(current_mappings.keys())
                remaining_labels = all_labels - set(final_mappings.keys())
                
                if not remaining_labels:
                    break
                    
                # Create new pairs list with only unmapped labels
                remaining_pairs = [(loc, label) for loc, label in location_label_pairs 
                                 if label in remaining_labels]
                
                print(f"Attempt {attempt + 1}: Mapped {len(mapped_labels)} labels, "
                      f"{len(remaining_labels)} remaining")
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                continue
        
        # Mark any remaining unmapped labels as "other"
        for label in all_labels:
            if label not in final_mappings:
                final_mappings[label] = "other"
                print(f"Warning: Label '{label[:50]}...' marked as 'other' after {max_attempts} attempts")
        
        return final_mappings
    
    def humanize_labels_without_location(self, labels: List[str]) -> Dict[str, str]:
        """
        Convert ALL long labels into unique, human-interpretable ones with verification and retry logic.
        
        Args:
            labels: List of long_label strings
            
        Returns:
            Dictionary mapping original long labels to human-interpretable labels
        """
        if not labels:
            return {}
        
        # Get all unique labels that need mapping
        all_labels = set(labels)
        final_mappings = {}
        remaining_pairs = labels.copy()
        max_attempts = 10
        
        for attempt in range(max_attempts):
            if not remaining_pairs:
                break
                
            prompt = self.create_prompt(remaining_pairs)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                )
                
                results = json.loads(response.choices[0].message.content)
                current_mappings = results.get("label_mappings", {})
                
                # Add successful mappings to final results
                final_mappings.update(current_mappings)
                
                # Find which labels still need mapping
                mapped_labels = set(current_mappings.keys())
                remaining_labels = all_labels - set(final_mappings.keys())
                
                if not remaining_labels:
                    break
                    
                # Create new pairs list with only unmapped labels
                remaining_pairs = [label for label in remaining_labels]
                
                print(f"Attempt {attempt + 1}: Mapped {len(mapped_labels)} labels, "
                      f"{len(remaining_labels)} remaining")
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                continue
        
        # Mark any remaining unmapped labels as "other"
        for label in all_labels:
            if label not in final_mappings:
                final_mappings[label] = "other"
                print(f"Warning: Label '{label[:50]}...' marked as 'other' after {max_attempts} attempts")
        
        return final_mappings

    def _create_prompt(self, location_label_pairs: List[Tuple[str, str]]) -> str:
        """Create user prompt with only the variable activity labels data."""
        
        # Remove duplicates and group labels by location for better organization
        unique_labels = set()
        location_groups = {}
        for location, label in location_label_pairs:
            if location not in location_groups:
                location_groups[location] = []
            if label not in unique_labels:
                location_groups[location].append(label)
                unique_labels.add(label)
        
        # Only variable content in user prompt for optimal caching
        labels_text = "ACTIVITY LABELS BY LOCATION:"
        label_index = 1
        for location, labels in location_groups.items():
            if labels:  # Only add location section if it has labels
                location_name = location.replace('_', ' ').title()
                labels_text += f"\n\n=== {location_name} ===\n"
                for label in labels:
                    labels_text += f"{label_index}. {label}\n"
                    label_index += 1
        
        return labels_text
    
    def create_prompt(self, labels: List[str]) -> str:
        """Create user prompt with only the variable activity labels data."""
        labels_text = "ACTIVITY LABELS:"
        label_index = 1
        for label in labels:
            labels_text += f"{label_index}. {label}\n"
            label_index += 1

        return labels_text


# Example usage
if __name__ == "__main__":
    # Initialize the humanizer
    
    # # Example location-label pairs
    # activity_pairs = [
    #     ("kitchen_sink_area", "chopping_and_slicing_produce_for_food_prep_near_sink"),
    #     ("kitchen_sink_area", "drinking_and_handling_beverages_near_sink OR placing_dishes_and_objects_into_sink_or_drying_rack"),
    #     ("kitchen_sink_area", "washing_and_rinsing_cups_bowls_and_containers_at_sink OR washing_and_rinsing_dishes_and_utensils_at_sink OR washing_and_rinsing_miscellaneous_objects_at_sink"),
    #     ("stovetop_cooking_zone", "flipping_and_stirring_food_in_pan_stovetop OR handling_and_moving_pans_on_stovetop OR turning_stovetop_burner_knobs"),
    #     ("refrigerator_area", "closing_refrigerator_door OR opening_refrigerator_door_for_item_access OR retrieving_items_from_refrigerator"),
    # ]
    
    # # Get human-interpretable labels
    # results = humanizer.humanize_labels(activity_pairs)
    
    # print("RESULTS:")
    # print("=" * 60)
    # for original, human in results.items():
    #     print(f"'{human}' ← '{original[:50]}{'...' if len(original) > 50 else ''}'")

    # load from the ensemble_with_gt.csv
    import dotenv
    dotenv.load_dotenv()
    import argparse
    import dotenv
    dotenv.load_dotenv()
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    sys.path.append(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4.1")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    base_results_dir = os.environ.get("BASE_RESULTS_DIR", "./results")
    participants = ["eu2a2-data-collection"]
    session_prefixes = ["P12"]
    vlm_model = 'gpt-4.1'
    alpha = args.alpha
    prob_threshold = args.prob_threshold
    model_name = args.model_name
    humanizer = ActivityLabelHumanizer(model_name=model_name)

    for (participant, session_prefix) in zip(participants, session_prefixes):
        
        results_dir = f"{base_results_dir}/{participant}/leaveoneout_analysis"
        ensemble_with_gt_file = f"{results_dir}/ensemble_with_gt_{alpha}_{prob_threshold}_details.csv"
        if not os.path.exists(ensemble_with_gt_file):
            print(f"File {ensemble_with_gt_file} does not exist")
            continue
        df_results = pd.read_csv(ensemble_with_gt_file)
        unique_labels = df_results['normalized_gt'].values.tolist() + df_results['normalized_pred'].values.tolist()
        unique_labels = list(set(unique_labels))

        results_file = f"{results_dir}/ensemble_with_gt_{alpha}_{prob_threshold}_{model_name}_final_labels.json"
        if os.path.exists(results_file):
            results = json.load(open(results_file, 'r'))
        else:
            results = humanizer.humanize_labels_without_location(unique_labels)
            with open(results_file, 'w') as f:
                json.dump(results, f)
        print(f"Model: {model_name} | Participant: {participant} | Session Prefix: {session_prefix} | Alpha: {alpha} | Prob Threshold: {prob_threshold}")
        for label, human_label in results.items():
            print(f"'{human_label[:50]}{'...' if len(human_label) > 50 else ''}' ← '{label[:50]}{'...' if len(label) > 50 else ''}'")
        print(f"\n\n")
        # break
