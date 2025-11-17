from typing import List, Dict, Any
import json
from openai import OpenAI


class KitchenLocationClustering:
    def __init__(self, model_name: str="gpt-4.1"):
        """Initialize the clustering class.

        Args:
            model_name: Name of the LLM model to use
        """
        self.client = OpenAI()  # Will use credentials from environment variables
        self.model_name = model_name

    def cluster_locations(self, locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cluster kitchen locations based on potential activities.

        Args:
            locations: List of dictionaries with 'location' and 'count' keys

        Returns:
            List of merged location clusters with counts
        """
        # Create the prompt
        prompt = self._create_clustering_prompt(locations)

        # Get LLM response
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        # Parse results
        results = json.loads(response.choices[0].message.content)
        return results["merged_locations"]

    def _create_clustering_prompt(self, locations: List[Dict[str, Any]]) -> str:
        """Create the prompt for activity-based location clustering."""
        locations_str = json.dumps(locations, indent=2)

        prompt = f"""As an AI agent analyzing kitchen activities, cluster the following kitchen locations based on the types of activities they support:

{locations_str}

Task: Create clusters of kitchen locations that represent distinct activity zones, following these guidelines:

1. Think about what activities can happen in each location
2. Merge locations that support the same types of kitchen activities.
3. Keep the appliance locations like, fridge, stove, and coffee machine in a separate cluster. 
3. All sitting spaces (chairs, tables, dining areas) should be merged as they represent meal consumption spaces
4. Keep cooking zones (stove, counter) separate as they represent different cooking activities
5. Storage areas (cabinets, shelves) in the same region should be merged
6. Only include locations with significant data (total count > 20).
7. Discard locations with very few counts that don't provide meaningful activity information

For example:
- "chair", "dining_table", "eating_area" should merge as they're all meal consumption spaces
- "stove_area", "stove_top", "near_stove" might merge as they represent the same cooking zone
- Different cabinet locations might merge if they're in the same kitchen region

Provide output in the following JSON format with example values:
{{
        "merged_locations": [
        {{
        "name": "dining_area",
            "total_count": 60,
            "original_locations": ["chair", "dining_table", "eating_area"],
            "supported_activities": ["eating", "dining", "having meals"]
        }},
        {{
        "name": "cabinet_storage",
            "total_count": 55,
            "original_locations": ["cabinet", "cabinet_area", "upper_cabinet"],
            "supported_activities": ["retrieving items", "storing items"]
        }}
    ]
}}

The format must be exact, using lower case with underscores for location names.
Include a list of supported activities for each merged location to show its functional purpose."""
        return prompt

    def match_multiple_locations(self, locations: List[str], merged_clusters: List[str]) -> Dict[str, Any]:
        """
        Match a new location string with existing merged clusters.

        Args:
            locations: New location strings to match
            merged_clusters: List of existing merged location clusters

        Returns:
            Dictionary with best matching cluster and confidence score
        """
        # Create the prompt
        clusters_str = json.dumps(merged_clusters, indent=2)

        prompt = f"""As an AI agent analyzing kitchen activities, determine which activity zone best matches these new locations:

    New locations: "{json.dumps(locations,indent=2)}"

    Existing activity zones:
    {clusters_str}

    Consider:
    1. What activities are possible in a new location
    2. How similar it is to existing zones
    3. Whether it represents the same functional area as any existing zone
    4. Only output a match if it actually is a really good fit, otherwise return "no_match"

    Provide output in the following JSON format:
    {{
        "location1": <best matching activity zone> or "no match",
        "location2": <best matching activity zone> or "no match",
    }}
"""
        # Get LLM response
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        # Parse and return results
        return json.loads(response.choices[0].message.content)

    def match_location(self, location: str, merged_clusters: List[str]) -> Dict[str, Any]:
            """
            Match a new location string with existing merged clusters.

            Args:
                location: New location string to match
                merged_clusters: List of existing merged location clusters

            Returns:
                Dictionary with best matching cluster and confidence score
            """
            # Create the prompt
            clusters_str = json.dumps(merged_clusters, indent=2)

            prompt = f"""As an AI agent analyzing kitchen activities, determine which activity zone best matches this new location:

        New location: "{location}"

        Existing activity zones:
        {clusters_str}

        Consider:
        1. What activities are possible in this new location
        2. How similar it is to existing zones
        3. Whether it represents the same functional area as any existing zone
        4. Only output a match if it actually is a really good fit, otherwise return "no_match"

        Provide output in the following JSON format:
        {{
            "matched_cluster": <best matching cluster>
        }}

        If no good match exists, return this:
        {{
            "matched_cluster": "no_match"
        }}
    """

            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            # Parse and return results
            return json.loads(response.choices[0].message.content)["matched_cluster"]

# Example usage
if __name__ == "__main__":
    # Sample data
    locations = [
        {"location": "dining_table", "count": 25},
        {"location": "chair", "count": 15},
        {"location": "eating_area", "count": 20},
        {"location": "cabinet", "count": 30},
        {"location": "cabinet_area", "count": 25},
        {"location": "stove_area", "count": 40},
    ]

    # Initialize clustering
    clustering = KitchenLocationClustering(model_name="gpt-4o")

    # Get clusters
    results = clustering.cluster_locations(locations)
    print(json.dumps(results, indent=2))