from typing import Dict, List, Tuple, Optional, Union
import json
from openai import OpenAI
from collections import defaultdict
import re
import ast


class LocationActivityClustering:
    def __init__(self, model_name: str = "gpt-4.1"):
        """
        Initialize with model name.

        Args:
            model_name: Name of the language model to use
        """
        self.client = OpenAI()
        self.model_name = model_name

    def parse_activity_string(self, activity_str: str) -> Dict[str, List[str]]:
        """
        Parse a formatted activity string into actions and objects.

        Args:
            activity_str: String in format "Actions: [...], Objects: [...]"

        Returns:
            Dictionary with 'actions' and 'objects' lists
        """
        try:
            # Extract lists using regex
            actions_match = re.search(r"Actions: (\[.*?\])", activity_str)
            objects_match = re.search(r"Objects: (\[.*?\])", activity_str)

            # Parse the lists using ast.literal_eval for safety
            actions = ast.literal_eval(actions_match.group(1)) if actions_match else []
            objects = ast.literal_eval(objects_match.group(1)) if objects_match else []

            return f"Actions: {actions}, Objects: {objects}"
        except Exception as e:
            print(f"Error parsing activity string: {str(e)}")
            return {'actions': [], 'objects': []}

    def cluster_activities(self, location: str, activity_strings: List[str]) -> Dict[str, List[str]]:
        """
        Cluster activities based on formatted strings containing actions and objects.

        Args:
            location: Kitchen location (e.g., "counter", "sink")
            activity_strings: List of strings in format "Actions: [...], Objects: [...]"

        Returns:
            Dictionary mapping merged activity names to lists of original activities
        """

        # split the activity string
        # Create prompt for the LLM
        prompt = self._create_cluster_prompt(location, sorted(activity_strings))

        try:
            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            results = json.loads(response.choices[0].message.content)
            return results

        except Exception as e:
            print(f"Error clustering activities: {str(e)}")
            return {}

    def match_activity(self, activity: str, merged_activities: List[str]) -> Dict[str, str]:
        """
        Match a new activity to existing merged activity clusters.

        Args:
            activity: Activity string in format "Actions: [...], Objects: [...]"
            merged_activities: List of existing merged activity names

        Returns:
            Dictionary with matched cluster or "no_match"
        """
        # Parse the new activity
        parsed_activity = self.parse_activity_string(activity)
        prompt = self._create_matching_prompt(parsed_activity, merged_activities)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"Error matching activity: {str(e)}")
            return {"matched_cluster": "no_match"}

    def _create_cluster_prompt(self, location: str, parsed_activities: List[str]) -> str:
        """Create prompt for clustering activities with action-object awareness."""
        # Format activities for display
        activities_list = "\n".join(parsed_activities)
        return f"""Given a list of kitchen activities (with their actions and objects) observed at the {location}, cluster them such that functionally similar activities are grouped together while preserving task-specific distinctions.

Activity Zone: {location}

Here are the observed activities:
-------
{activities_list}
-------

Guidelines:
1. Preserve Task-Specific Distinctions:
   - Keep activities related to different recipes or meal preparations separate
   - For example: "pouring cereal" and "pouring coffee" should be separate clusters as they relate to different tasks
   - Maintain distinction between hot and cold beverage preparations
   - Preserve differences between meal components (e.g., main dish vs. condiments)

2. Clustering Criteria:
   - Group activities based on both action AND purpose/context
   - Consider the typical sequence of tasks the activity belongs to
   - Activities using similar objects for the same purpose can be grouped
   - Drop activities unrelated to kitchen tasks or not relevant to this location
   - Drop activities that involve user movement or are not specific to the location
   - Drop miscellaneous activities.
   - Any activities separated by a "or" would mean that it can be any of the activities on the list, try to name it something higher level than those activities if it is possible.

3. Cluster Naming:
   - Use specific, descriptive names that capture both action and context
   - Include key object categories in cluster names when relevant
   - Example: "preparing_cereal_bowl" vs "preparing_coffee_drink"

Keep the count of clusters as MINIMAL possible while maintaining meaningful distinctions.

Return a JSON array containing the clusters and their *TOP FEW* matching original activities:
{{
"[specific_action]_[context]": ["original_activity1", "original_activity2", ...],
...
}}

Focus on creating meaningful, task-specific clusters that help distinguish between different kitchen activities while avoiding overly generic groupings."""

    def _create_matching_prompt(self, parsed_activity: Dict[str, List[str]],
                                merged_activities: List[str]) -> str:
        """Create prompt for matching a new activity."""
        activities_list = "\n".join([f"- {act}" for act in merged_activities])

        actions_str = ", ".join(parsed_activity['actions']) if parsed_activity['actions'] else "no specific action"
        objects_str = ", ".join(parsed_activity['objects']) if parsed_activity['objects'] else "no specific objects"

        return f"""Given a new kitchen activity with its actions and objects, match it to the most relevant existing merged activity cluster based on similar interaction patterns.

New Activity:
Actions: {actions_str}
Objects: {objects_str}

Existing Merged Activities:
{activities_list}

Guidelines:
1. Consider both actions and objects when matching:
   - Similar action patterns
   - Related object interactions
   - Comparable manipulation sequences
2. Look for matching interaction patterns:
   - Similar object manipulations
   - Related action sequences
   - Common usage patterns
3. Return "no_match" if:
   - Action patterns are significantly different
   - Object interactions don't align
   - The overall interaction pattern is distinct

Return response in this JSON format:
{{
    "matched_cluster": <merged_activity_name or "no_match">
}}"""

    def summarize_clusters(self, mappings: Dict[str, List[str]],
                           activity_strings: List[str]) -> Dict[str, int]:
        """
        Summarize clustered activities by counting occurrences.

        Args:
            mappings: Dictionary mapping merged activities to lists of original activities
            activity_strings: Original list of activity strings

        Returns:
            Dictionary mapping merged activities to total counts
        """
        summary = defaultdict(int)

        # Create a mapping from activity index to cluster
        activity_to_cluster = {}
        for cluster_name, activities in mappings.items():
            for activity in activities:
                # Extract index from "Activity X" format
                idx = int(activity.split()[1]) - 1
                activity_to_cluster[idx] = cluster_name

        # Count occurrences
        for idx in activity_to_cluster:
            if idx < len(activity_strings):
                cluster_name = activity_to_cluster[idx]
                summary[cluster_name] += 1

        return dict(summary)

    def match_activities_batch(self, activities: List[str], merged_activities: List[str]) -> Dict[str, str]:
        """
        Match multiple new activities to existing merged activity clusters in a single request.

        Args:
            activities: List of activity strings in format "Actions: [...], Objects: [...]"
            merged_activities: List of existing merged activity names

        Returns:
            Dictionary mapping original activity strings to matched cluster names
        """
        # # Parse all activities
        # parsed_activities = [self.parse_activity_string(activity) for activity in activities]
        #
        # # Create batch matching prompt
        # activities_list = ""
        # for i, (activity, parsed) in enumerate(zip(activities, parsed_activities), 1):
        #     actions_str = ", ".join(parsed['actions']) if parsed['actions'] else "no specific action"
        #     objects_str = ", ".join(parsed['objects']) if parsed['objects'] else "no specific objects"
        #     activities_list += f"Activity {i}:\n  Actions: {actions_str}\n  Objects: {objects_str}\n"
        activities_list = "\n".join(activities)
        clusters_list = "\n".join([f"- {act}" for act in merged_activities])

        batch_prompt = f"""Match each of these new kitchen activities to the most relevant existing merged activity cluster based on similar interaction patterns.

New Activities to Match:
{activities_list}

Existing Merged Activity Clusters:
{clusters_list}

Guidelines:
1. Consider both actions and objects when matching:
   - Similar action patterns
   - Related object interactions
   - Comparable manipulation sequences
2. Look for matching interaction patterns:
   - Similar object manipulations
   - Related action sequences
   - Common usage patterns
3. Use "no_match" if:
   - Action patterns are significantly different
   - Object interactions don't align
   - The overall interaction pattern is distinct

Return a JSON object mapping each input activity to its matched cluster:
{{
    "Activity 1": <merged_activity_name or "no_match">,
    "Activity 2": <merged_activity_name or "no_match">,
    ...
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": batch_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            # Create mapping from response to original activities
            results = json.loads(response.choices[0].message.content)
            return results

        except Exception as e:
            print(f"Error batch matching activities: {str(e)}")
            return {activity: "no_match" for activity in activities}

    def consolidate_location_actions(self, activity_strings: List[str], location: str) -> Dict[str, str]:
        """Consolidate location actions based on similar interaction patterns."""
        # Parse and format activities
        activities_list = ""
        for i, activity_str in enumerate(activity_strings, 1):
            parsed = self.parse_activity_string(activity_str)
            actions_str = ", ".join(parsed['actions']) if parsed['actions'] else "no specific action"
            objects_str = ", ".join(parsed['objects']) if parsed['objects'] else "no specific objects"
            activities_list += f"Activity {i}:\n  Actions: {actions_str}\n  Objects: {objects_str}\n"

        consolidation_prompt = f"""Given these activities at {location}:
{activities_list}

Consolidate similar activities based on their interaction patterns, following these rules:
- Consider both actions and objects when consolidating
- Only combine activities with very similar interaction patterns
- Keep distinct patterns separate
- Use "/" to combine nearly identical patterns
- Use clear, descriptive labels that capture the interaction

Return format:
{{
   "Activity 1": "new_label1",
   "Activity 2": "new_label1", 
   "Activity 3": "new_label2"
}}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": consolidation_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error consolidating activities: {str(e)}")
            return {}



# Example usage
if __name__ == "__main__":
    # Initialize clusterer
    clusterer = LocationActivityClustering(model_name="gpt-4")

    # Example activities at counter
    counter_activities = [
        "Actions: ['placing_kettle_on_counter'], Objects: ['kettle']",
        "Actions: ['opening_drawer', 'reaching_into_cabinet'], Objects: ['bag_of_items', 'drawer_handle']",
        "Actions: ['pouring_content_into_cup'], Objects: ['container', 'cup']",
        "Actions: ['opening_container', 'pouring_contents_into_cup'], Objects: ['container', 'packet', 'small_cup']",
        "Actions: ['grabbing_coffee_pod', 'moving_towards_coffee_machine'], Objects: []",
        "Actions: ['standing, interacting with counter items'], Objects: ['coffee cup']"
    ]

    # Cluster activities
    mappings = clusterer.cluster_activities("counter", counter_activities)

    print("\nActivity Clusters:")
    for merged, originals in mappings.items():
        print(f"\n{merged}:")
        for orig in originals:
            print(f"  - {orig}")

    # Get summary
    summary = clusterer.summarize_clusters(mappings, counter_activities)

    print("\nClustered Summary:")
    for activity, count in summary.items():
        print(f"{activity}: {count}")

    # Example of matching new activity
    new_activity = "Actions: ['inserting_pod', 'pressing_button'], Objects: ['coffee_machine', 'coffee_pod']"
    merged_activities = list(mappings.keys())
    match_result = clusterer.match_activity(new_activity, merged_activities)

    print(f"\nMatching new activity:")
    print(f"Input: {new_activity}")
    print(f"Matched to: {match_result['matched_cluster']}")

    # Example of batch matching multiple activities
    print("\nBatch Matching Activities:")
    new_activities = [
        "Actions: ['inserting_pod', 'pressing_button'], Objects: ['coffee_machine', 'coffee_pod']",
        "Actions: ['pouring_water', 'placing_kettle'], Objects: ['kettle', 'counter']",
        "Actions: ['stirring_contents'], Objects: ['cup', 'spoon']"
    ]
    batch_results = clusterer.match_activities_batch(new_activities, merged_activities)

    print("\nBatch Matching Results:")
    for activity, matched_cluster in batch_results.items():
        print(f"\nActivity: {activity}")
        print(f"Matched to: {matched_cluster}")
