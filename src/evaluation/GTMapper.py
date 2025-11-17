from typing import List, Dict, Tuple, Optional, Any
from openai import OpenAI
import json
import os
from datetime import datetime
import time
import re


class GTMapper:
    def __init__(self, model: str = "gpt-4.1", cache_dir: str = "cache"):
        """Initialize the mapper with OpenAI credentials and cache directory."""
        self.client = OpenAI()
        self.model = model
        self.cache_dir = cache_dir

        # Create cache directories
        self.batch_dir = os.path.join(cache_dir, 'batches')
        self.results_dir = os.path.join(cache_dir, 'results')
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def map_ground_truth(self, discovered_labels: List[str], ground_truth: str) -> Dict:
        """
        Map a single ground truth label to discovered activity labels.

        Args:
            discovered_labels: List of discovered label groups
            ground_truth: The ground truth label to map

        Returns:
            Dictionary with mapping results
        """
        prompt = self._create_prompt(discovered_labels, ground_truth)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"Error in mapping ground truth: {e}")
            return {"error": str(e)}

    def create_batch(self,
                     discovered_labels: List[str],
                     ground_truth_labels: List[str],
                     batch_id: Optional[str] = None,
                     max_batch_size_mb: int = 95) -> str:
        """
        Create a batch request for processing multiple ground truth labels.

        Args:
            discovered_labels: List of discovered label groups
            ground_truth_labels: List of ground truth labels to map
            batch_id: Optional custom batch ID
            max_batch_size_mb: Maximum batch file size in MB

        Returns:
            Batch ID string
        """
        # Generate batch ID if not provided
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Check if batch already exists
        if self.batch_exists(batch_id):
            print(f"Batch {batch_id} already exists")
            return batch_id

        batch_request_file = os.path.join(self.batch_dir, f"{batch_id}_request.jsonl")
        num_requests = 0

        try:
            with open(batch_request_file, 'w') as f:
                for idx, gt_label in enumerate(ground_truth_labels):
                    gt_id = f"gt_{idx}"
                    cache_key = f"{batch_id}___{idx}___{gt_id}"

                    # Skip if result already cached
                    if self._load_from_cache(f"{cache_key}_result.json"):
                        continue

                    prompt = self._create_prompt(discovered_labels, gt_label)
                    request_data = {
                        "custom_id": cache_key,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.0
                        }
                    }

                    # Store original ground truth for later use
                    metadata = {
                        "ground_truth": gt_label,
                        "gt_id": gt_id
                    }
                    self._save_to_cache(f"{cache_key}_metadata.json", metadata)

                    f.write(json.dumps(request_data) + '\n')
                    num_requests += 1

            # Check batch file size
            if os.path.getsize(batch_request_file) > max_batch_size_mb * 1024 * 1024:
                raise ValueError(f"Batch request file size exceeds {max_batch_size_mb} MB")

            if num_requests > 0:
                batch_response = self._send_batch_to_openai(batch_request_file)
                self._save_to_cache(f"{batch_id}_response.json", json.loads(batch_response.json()))
                print(f"Created batch {batch_id} with {num_requests} requests")
            else:
                print("No new requests to process")

            return batch_id

        except Exception as e:
            print(f"Error creating batch {batch_id}: {str(e)}")
            if os.path.exists(batch_request_file):
                os.remove(batch_request_file)
            raise

    def get_batch_results(self,
                          batch_id: str,
                          wait_for_completion: bool = False,
                          max_wait_time: int = 3600) -> Tuple[Optional[Dict], str]:
        """
        Get results for a batch request.

        Args:
            batch_id: Batch identifier
            wait_for_completion: Whether to wait for batch completion
            max_wait_time: Maximum wait time in seconds

        Returns:
            Tuple of (results dict, status string)
        """
        start_time = time.time()

        while True:
            try:
                batch_response = self._fetch_batch_response(batch_id)

                if batch_response['status'] == "failed":
                    print(f"Batch {batch_id} failed: {batch_response.get('errors', 'Unknown error')}")
                    return None, "failed"

                if batch_response['status'] == "completed":
                    return self._process_completed_batch(batch_id, batch_response), "completed"

                if not wait_for_completion:
                    return None, batch_response['status']

                if time.time() - start_time > max_wait_time:
                    print(f"Timeout waiting for batch {batch_id}")
                    return None, "timeout"

                print(f"Batch {batch_id} status: {batch_response['status']}. Waiting...")
                time.sleep(10)

            except Exception as e:
                print(f"Error getting results for batch {batch_id}: {str(e)}")
                return None, "error"

    def batch_exists(self, batch_id: str) -> bool:
        """Check if a batch already exists."""
        response_file = os.path.join(self.results_dir, f"{batch_id}_response.json")
        return os.path.exists(response_file)

    def _create_prompt(self, discovered_labels: List[str], ground_truth: str) -> str:
        """Create the prompt for the language model."""
        prompt = f"""You are an expert in activity recognition systems. You need to map ground truth activity labels to discovered activity labels for kitchen activities.

DISCOVERED ACTIVITY LABELS (each group represents alternatives where any one activity could be happening):
{json.dumps(discovered_labels, indent=2)}

GROUND TRUTH LABEL TO MAP:
{ground_truth}

TASK:
1. Analyze the ground truth label and identify its key components.
2. If the ground truth label contains multiple activities connected by "AND" as well as "/", you must map EACH component to the appropriate discovered label group(s).
3. For each ground truth activity (or component of a compound activity), find the most semantically similar discovered label group.
4. The final mapping should be the UNION of all matched discovered label groups needed to cover the entire ground truth label.
5. Provide your reasoning for each mapping decision.

Remember:
- Each discovered label group represents alternatives (separated by "OR" usually) where any activity in that group could be happening
- Prioritize semantic similarity over exact word matching
- Choose the most specific match when possible
- If no exact match exists, do not force a mapping.
- The final mapping should include ALL discovered label groups needed to cover every component in the ground truth label

Return your analysis in this JSON format:
{{
    "ground_truth": "{ground_truth}",
    "mapping": ["discovered_label_group1", "discovered_label_group2", ...],
    "reasoning": "Detailed step-by-step explanation of your mapping process, explaining each component"
}}

Your response should be valid JSON only, with no preamble or conclusion text.
"""
        return prompt

    def _send_batch_to_openai(self, batch_file: str) -> Any:
        """Send batch request to OpenAI API."""
        with open(batch_file, "rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")

        return self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

    def _process_completed_batch(self, batch_id: str, batch_response: Dict) -> Dict[str, Any]:
        """Process a completed batch and return results."""
        try:
            batch_results = self.client.files.content(batch_response['output_file_id'])
            results = {}

            for line in batch_results.text.strip().split('\n'):
                response = json.loads(line)
                custom_id = response['custom_id']
                content = response['response']['body']['choices'][0]['message']['content']

                try:
                    # Load the original metadata
                    metadata = self._load_from_cache(f"{custom_id}_metadata.json")
                    if not metadata:
                        print(f"Warning: No metadata found for {custom_id}")
                        continue

                    # Parse the LLM response
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    parsed_result = json.loads(json_match.group() if json_match else content)

                    # Combine metadata with results
                    combined_result = {
                        "gt_id": metadata.get("gt_id", "unknown"),
                        "ground_truth": metadata.get("ground_truth", ""),
                        "mapping": parsed_result.get("mapping", []),
                        "reasoning": parsed_result.get("reasoning", "")
                    }

                    results[custom_id] = combined_result
                    self._save_to_cache(f"{custom_id}_result.json", combined_result)
                except Exception as e:
                    print(f"Error parsing result for {custom_id}: {str(e)}")
                    continue

            return results

        except Exception as e:
            print(f"Error processing batch {batch_id}: {str(e)}")
            return {}

    def _fetch_batch_response(self, batch_id: str) -> Dict[str, Any]:
        """Fetch batch response from cache or API."""
        response_file = os.path.join(self.results_dir, f"{batch_id}_response.json")
        if not os.path.exists(response_file):
            raise FileNotFoundError(f"Batch response file not found: {batch_id}")

        with open(response_file, 'r') as f:
            existing_response = json.load(f)

        openai_batch_id = existing_response['id']
        return json.loads(self.client.batches.retrieve(batch_id=openai_batch_id).json())

    def _save_to_cache(self, cache_file: str, data: Any) -> None:
        """Save data to cache file."""
        with open(os.path.join(self.results_dir, cache_file), 'w') as f:
            json.dump(data, f)

    def _load_from_cache(self, cache_file: str) -> Any:
        """Load data from cache file."""
        cache_path = os.path.join(self.results_dir, cache_file)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def compile_all_results(self, batch_id: str) -> Dict[str, Any]:
        """
        Compile all results from a batch into a structured format.

        Args:
            batch_id: Batch identifier

        Returns:
            Dictionary with all mapping results
        """
        try:
            # Get all result files for this batch
            result_files = [f for f in os.listdir(self.results_dir)
                            if f.startswith(batch_id) and f.endswith('_result.json')]

            compiled_results = {
                "batch_id": batch_id,
                "timestamp": datetime.now().isoformat(),
                "total_mappings": len(result_files),
                "mappings": []
            }

            for result_file in result_files:
                result_path = os.path.join(self.results_dir, result_file)
                with open(result_path, 'r') as f:
                    result = json.load(f)

                mapping = {
                    "gt_id": result.get("gt_id", "unknown"),
                    "ground_truth": result.get("ground_truth", ""),
                    "mapping": result.get("mapping", []),
                    "reasoning": result.get("reasoning", "")
                }
                compiled_results["mappings"].append(mapping)

            return compiled_results

        except Exception as e:
            print(f"Error compiling results for batch {batch_id}: {str(e)}")
            return {"error": str(e), "mappings": []}