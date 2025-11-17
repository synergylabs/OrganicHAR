import os
import glob
import json
import time
import cv2
import base64
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
from openai import OpenAI
from litellm import completion

class SegmentDescriptionGenerator:
    """
    Simple class to batch process video snippets from a directory.
    Takes a directory of video files and processes them in batches.
    """

    def __init__(self, vlm_model: str, cache_dir: str, batch_size: int = 20):
        """
        Initialize the segment description generator.
        
        Args:
            vlm_model: OpenAI model to use for analysis
            cache_dir: Directory to store cache and batch results
            batch_size: Number of videos per batch
        """
        self.client = OpenAI()
        self.vlm_model = vlm_model
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        
        # Create directories
        self.batch_dir = os.path.join(cache_dir, 'segment_batches')
        self.results_dir = os.path.join(cache_dir, 'results')
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def process_directory(self, video_dir: str, 
                         file_extensions: List[str] = None,
                         processing_id: str = None,
                         exisiting_video_paths: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process all video snippets in a directory.
        
        Args:
            video_dir: Directory containing video snippets
            file_extensions: List of video file extensions to process (default: ['*.mp4', '*.MOV', '*.avi'])
            processing_id: Optional ID for this processing session
            exisiting_video_paths: Optional dictionary of existing results to skip
        Returns:
            Dictionary with results for each video snippet
        """
        if file_extensions is None:
            file_extensions = ['*.mp4', '*.MOV', '*.avi']
            
        if processing_id is None:
            processing_id = f"segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Find all video files
        video_paths = []
        for ext in file_extensions:
            video_paths.extend(glob.glob(os.path.join(video_dir, ext)))
            video_paths.extend(glob.glob(os.path.join(video_dir, '**', ext), recursive=True))
        
        video_paths = sorted(list(set(video_paths)))  # Remove duplicates and sort

        #remove the duplicates where the file basename is the same
        video_file_names = [os.path.basename(path) for path in video_paths]
        video_paths = [path for i, path in enumerate(video_paths) if video_file_names[i] not in video_file_names[:i]]

        #remove the videos that are already in the existing results
        video_paths = [path for path in video_paths if os.path.basename(path) not in exisiting_video_paths]

        video_paths = sorted(list(set(video_paths)))  # Remove duplicates and sort
        
        if not video_paths:
            print(f"No video files found in {video_dir}")
            return {}

        print(f"Found {len(video_paths)} video files to process")
        
        # Create batches
        batch_info = self._create_batches(video_paths, processing_id)
        
        # Save processing info
        self._save_processing_info(processing_id, {
            'video_dir': video_dir,
            'video_paths': video_paths,
            'batch_ids': batch_info['batch_ids'],
            'creation_time': datetime.now().isoformat(),
            'total_videos': len(video_paths)
        })
        
        print(f"Created {len(batch_info['batch_ids'])} batches for processing")
        return {'processing_id': processing_id, **batch_info}

    def get_results(self, processing_id: str, 
                   max_wait_time: int = 300,
                   check_interval: int = 10) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get results from all batches for a processing session.
        
        Args:
            processing_id: The processing session identifier
            max_wait_time: Maximum time to wait for results (seconds)
            check_interval: How often to check for completion (seconds)
            
        Returns:
            Tuple of (results, status_info)
        """
        processing_info = self._load_processing_info(processing_id)
        if not processing_info:
            raise ValueError(f"No processing information found for ID: {processing_id}")

        batch_ids = processing_info['batch_ids']
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            results = {}
            status = {
                'completed': [],
                'pending': [],
                'failed': [],
                'total_batches': len(batch_ids),
                'completion_rate': 0.0
            }

            for batch_id in batch_ids:
                try:
                    batch_results, batch_status = self._get_batch_results(batch_id)
                    
                    if batch_status == "completed":
                        results.update(batch_results)
                        status['completed'].append(batch_id)
                    elif batch_status == "failed":
                        status['failed'].append(batch_id)
                    else:
                        status['pending'].append(batch_id)
                        
                except Exception as e:
                    print(f"Error checking batch {batch_id}: {str(e)}")
                    status['failed'].append(batch_id)

            status['completion_rate'] = len(status['completed']) / status['total_batches'] * 100
            
            # If all batches are done (completed or failed), return results
            if not status['pending']:
                return results, status
                
            # Otherwise, wait and check again
            print(f"Progress: {status['completion_rate']:.1f}% complete "
                  f"({len(status['completed'])}/{status['total_batches']} batches)")
            time.sleep(check_interval)
        
        # Timeout reached
        print(f"Timeout reached after {max_wait_time} seconds")
        return results, status

    def process_single_video(self, video_path: str, frames_per_second: float = 1.0) -> Dict[str, Any]:
        """
        Process a single video file synchronously and return the result immediately.
        
        Args:
            video_path: Path to the video file to process
            frames_per_second: Frame extraction rate (default: 1.0)
            
        Returns:
            Dictionary with the analysis result for the video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract frames from video
        frames = self._preprocess_video(video_path, frames_per_second)
        if not frames:
            return {
                "error": "Could not extract frames from video",
                "location_analysis": [{
                    "primary_location": "",
                    "action_at_location": "",
                    "user_movement": "",
                    "confidence": 0.0
                }]
            }
        
        # Create prompt
        prompt = self._create_prompt(frames)
        
        # Make synchronous API call
        try:
            response = self.client.chat.completions.create(
                model=self.vlm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            # Extract and parse the response
            response_content = response.choices[0].message.content
            result = self._parse_json_response(response_content)
            
            return result
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return {
                "error": str(e),
                "location_analysis": [{
                    "primary_location": "",
                    "action_at_location": "",
                    "user_movement": "",
                    "confidence": 0.0
                }]
            }

    def _create_batches(self, video_paths: List[str], processing_id: str) -> Dict[str, Any]:
        """Create batches from video paths."""
        batch_ids = []
        
        # Split videos into batches
        for i in range(0, len(video_paths), self.batch_size):
            batch_paths = video_paths[i:i + self.batch_size]
            batch_id = f"{processing_id}_batch_{i // self.batch_size:03d}"
            
            try:
                # Check if batch already exists
                if not self._batch_exists(batch_id):
                    batch_id = self._create_batch_request(
                        video_paths=batch_paths,
                        frames_per_second=1.0,
                        batch_id=batch_id,
                        force_recompute=False
                    )
                
                batch_ids.append(batch_id)
                print(f"Created batch {batch_id} with {len(batch_paths)} videos")
                
            except Exception as e:
                print(f"Error creating batch {batch_id}: {str(e)}")
                continue
        
        return {'batch_ids': batch_ids}

    def _batch_exists(self, batch_id: str) -> bool:
        """Check if a batch already exists."""
        return os.path.exists(os.path.join(self.results_dir, f"{batch_id}_response.json"))

    def _create_batch_request(self, video_paths: List[str], frames_per_second: float = 1.0,
                             max_batch_size_in_mb: int = 95, batch_id: str = None, 
                             force_recompute: bool = True) -> str:
        """Create a batch request for OpenAI processing."""
        if batch_id is None:
            batch_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
        batch_request_file = os.path.join(self.batch_dir, f"{batch_id}_request.jsonl")
        num_requests = 0

        # Remove existing batch request file if forcing recompute
        if os.path.exists(batch_request_file):
            if force_recompute:
                os.remove(batch_request_file)
            else:
                return batch_id

        with open(batch_request_file, 'a+') as f:
            for video_path in video_paths:
                cache_key = self._get_cache_key(video_path, frames_per_second)
                if self._load_from_cache(f"{cache_key}_result.json"):
                    continue

                frames = self._get_preprocessed_frames(video_path, frames_per_second)
                if not frames:
                    continue
                    
                prompt = self._create_prompt(frames)
                request_data = self._create_request_data(batch_id, cache_key, prompt)
                f.write(json.dumps(request_data) + '\n')
                num_requests += 1

        # Check file size
        if num_requests == 0:
            print("No new requests added to batch, all results are already cached")
            return batch_id
        
        if os.path.getsize(batch_request_file) > max_batch_size_in_mb * 1024 * 1024:
            batch_file_size = os.path.getsize(batch_request_file) / 1024 / 1024
            raise ValueError(f"Batch request file size ({batch_file_size} MB) exceeds {max_batch_size_in_mb} MB")

        if num_requests == 0:
            print("No new requests added to batch, all results are already cached")
            return batch_id

        # Send batch to OpenAI
        
        batch_init_response = self._send_batch_to_openai(batch_request_file)
        self._save_to_cache(f"{batch_id}_response.json", json.loads(batch_init_response.json()))
        return batch_id

    def _get_preprocessed_frames(self, video_path: str, frames_per_second: float) -> List[str]:
        """Get preprocessed frames from video, with caching."""
        cache_key = self._get_cache_key(video_path, frames_per_second)
        cached_frames = self._load_from_cache(f"{cache_key}_frames.json")

        if cached_frames:
            return cached_frames

        frames = self._preprocess_video(video_path, frames_per_second)
        self._save_to_cache(f"{cache_key}_frames.json", frames)
        return frames

    def _preprocess_video(self, video_path: str, frames_per_second: float) -> List[str]:
        """Extract frames from video and encode as base64."""
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            video.release()
            return []
            
        frame_interval = max(1, int(fps / frames_per_second))
        frames = []
        frame_count = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # if depth frame of 1280 x 480, resize to 640 x 480 by keeping the left half
                if frame.shape[1] == 1280:
                    frame = frame[:, :640, :]
                _, buffer = cv2.imencode('.jpg', frame)
                img_str = base64.b64encode(buffer).decode('utf-8')
                frames.append(img_str)
                
            frame_count += 1
            
            # Limit to reasonable number of frames
            if len(frames) >= 50:
                break

        video.release()
        return frames

    def _create_prompt(self, frames: List[str]) -> List[Dict[str, Any]]:
        """Create prompt for activity detection."""
        prompt_text = """Analyze this kitchen activity video sequence and provide precise location-based activity analysis.

        FOCUS: Generate specific, detailed labels for WHERE actions happen and WHAT actions are performed.

        PRIMARY REQUIREMENTS:

        1. **Precise Location Identification**:
        - Be equipment/appliance specific: "coffee_machine", "kitchen_sink", "refrigerator", "stovetop_burner_1", "microwave", "dishwasher"
        - NOT generic terms like "kitchen", "counter", "appliance"
        - If at a counter, specify what's on it: "counter_near_coffee_machine", "island_prep_area"

        2. **Specific Action Description**:
        - Combine action + object interaction: "pressing_buttons on coffee_machine", "turning_knob on stovetop", "opening_door of refrigerator"
        - Be precise about the interaction method: "pressing", "turning", "pulling", "lifting", "pouring_into"
        - Include the target object: "on coffee_machine", "into sink", "from refrigerator"

        3. **Movement Analysis**:
        - "Stationary" - person staying in same location
        - "Moving to [specific_location]" - person transitioning between locations
        - Be specific about destination: "Moving to refrigerator", "Moving to sink_area"

        Return as JSON:
        {
            "location_analysis": [
                {
                    "primary_location": "coffee_machine",
                    "action_at_location": "pressing_buttons on coffee_machine",
                    "user_movement": "Stationary",
                    "confidence": 0.92
                }
            ]
        }

        OR if uncertain, provide multiple possibilities (max 3):
        {
            "location_analysis": [
                {
                    "primary_location": "coffee_machine",
                    "action_at_location": "pressing_buttons on coffee_machine", 
                    "user_movement": "Stationary",
                    "confidence": 0.65
                },
                {
                    "primary_location": "microwave",
                    "action_at_location": "opening_door of microwave",
                    "user_movement": "Stationary", 
                    "confidence": 0.55
                }
            ]
        }

        SPECIFICITY GUIDELINES:
        - Equipment-level precision for locations (not room-level)
        - Action-object interaction detail (not just the action)
        - Include directional/spatial information when relevant
        - Always provide at least one location analysis
        - If multiple locations involved, focus on where the primary action occurs

        MULTIPLE ANALYSES DECISION:
        - **Single Analysis**: Provide only one when confidence ≥0.8 (clear, definitive observation)
        - **Multiple Analyses**: Provide 2-3 options when confidence <0.8 for any individual interpretation
        - Rank analyses by confidence (highest first)
        - Each analysis should represent a genuinely different interpretation of the scene
        - Don't provide multiple analyses for minor variations (e.g., "left_burner" vs "right_burner")

        HANDLING UNCLEAR FRAMES:
        - If frames are blurry, dark, or partially obscured, use your best educated guess
        - Base inferences on visible context clues: kitchen layout, partial equipment visibility, body positioning
        - Use typical kitchen workflows to inform reasonable assumptions
        - When uncertain, provide multiple plausible interpretations with respective confidence scores
        - Examples of scenarios for multiple analyses:
        * Ambiguous appliance interaction → could be "coffee_machine" (0.6) or "microwave" (0.5)
        * Unclear counter activity → could be "food_prep on counter" (0.7) or "reaching_for items_in_cabinet" (0.6)
        * Person near multiple appliances → primary focus could be different locations
        - Always provide your best specific guess rather than generic terms like "unknown_appliance" or "unclear_location"
    """
        return [prompt_text, *[{"image": frame, "resize": 768} for frame in frames]]

    def _create_request_data(self, batch_id: str, cache_key: str, prompt: List) -> Dict:
        """Create request data for OpenAI batch API."""
        return {
            "custom_id": f"{batch_id}__{cache_key}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.vlm_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500
            }
        }

    def _send_batch_to_openai(self, batch_file: str):
        """Send batch file to OpenAI."""
        with open(batch_file, "rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")
        
        batch_response = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(f"Batch submitted: {batch_response.id}")
        return batch_response
    
    def _send_batch_to_gemini(self, batch_file: str):
        """Send batch file to Gemini."""
        pass

    def _send_batch_to_anthropic(self, batch_file: str):
        """Send batch file to Anthropic."""
        pass

    def _get_cache_key(self, video_path: str, frames_per_second: float) -> str:
        """Generate cache key for video."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        return f"activity_{video_name}_{frames_per_second}".replace('/', '_').replace('\\', '_')

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

    def _fetch_batch_response(self, batch_id: str):
        """Fetch batch response from OpenAI."""
        existing_response_file = os.path.join(self.results_dir, f"{batch_id}_response.json")
        if not os.path.exists(existing_response_file):
            raise FileNotFoundError(f"Batch response file not found for batch ID: {batch_id}")

        with open(existing_response_file, 'r') as f:
            existing_response = json.load(f)

        openai_batch_id = existing_response['id']
        return json.loads(self.client.batches.retrieve(batch_id=openai_batch_id).json())

    def _process_batch_results(self, batch_result_file: str) -> Dict[str, Any]:
        """Process batch results from OpenAI."""
        results = {}
        with open(batch_result_file, 'r') as f:
            for line in f:
                response = json.loads(line)
                video_key = response['custom_id'].split('__')[1]
                response_content = response['response']['body']['choices'][0]['message']['content']
                parsed_result = self._parse_json_response(response_content)
                results[video_key] = parsed_result
                self._save_to_cache(f"{video_key}_result.json", parsed_result)
        return results

    def _parse_json_response(self, response: Union[str, dict]) -> dict:
        """Parse JSON response from OpenAI."""
        if isinstance(response, dict):
            return response

        try:
            # Try to extract JSON from markdown code block
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            json_str = match.group(1) if match else response
            parsed_response = json.loads(json_str)

            # Validate the new format with location_analysis
            if "location_analysis" in parsed_response:
                # Ensure location_analysis is a list
                if not isinstance(parsed_response["location_analysis"], list):
                    parsed_response["location_analysis"] = [parsed_response["location_analysis"]]
                
                # Validate each analysis entry has required fields
                for analysis in parsed_response["location_analysis"]:
                    if not isinstance(analysis, dict):
                        continue
                    # Ensure required fields exist with defaults
                    analysis.setdefault("primary_location", "")
                    analysis.setdefault("action_at_location", "")
                    analysis.setdefault("user_movement", "")
                    analysis.setdefault("confidence", 0.0)
                
                return parsed_response
            else:
                # If location_analysis is missing, create default structure
                return {
                    "location_analysis": [{
                        "primary_location": "",
                        "action_at_location": "",
                        "user_movement": "",
                        "confidence": 0.0
                    }]
                }

        except Exception as e:
            print(f"Error parsing response: {e}")
            # Return default structure on error
            return {
                "location_analysis": [{
                    "primary_location": "",
                    "action_at_location": "",
                    "user_movement": "",
                    "confidence": 0.0
                }]
            }

    def _get_batch_results(self, batch_id: str):
        """Get results from a specific batch."""
        batch_result_file = os.path.join(self.results_dir, f"{batch_id}_result.jsonl")

        if not os.path.exists(batch_result_file):
            batch_response = self._fetch_batch_response(batch_id)
            if batch_response['status'] == "failed":
                print(f"Batch request failed: {batch_response.get('errors', 'Unknown error')}")
                return batch_response, batch_response['status']

            if not batch_response or batch_response['status'] != "completed":
                return None, batch_response['status']

            batch_results = self.client.files.content(batch_response['output_file_id'])
            with open(batch_result_file, 'w') as f:
                f.write(batch_results.text)

        return self._process_batch_results(batch_result_file), "completed"

    def _save_processing_info(self, processing_id: str, info: Dict[str, Any]) -> None:
        """Save processing session information."""
        path = os.path.join(self.batch_dir, f"{processing_id}_info.json")
        with open(path, 'w') as f:
            json.dump(info, f, indent=2)

    def _load_processing_info(self, processing_id: str) -> Optional[Dict[str, Any]]:
        """Load processing session information."""
        path = os.path.join(self.batch_dir, f"{processing_id}_info.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def print_results_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the analysis results."""
        if not results:
            print("No results to display")
            return
            
        print(f"\n=== Analysis Summary for {len(results)} videos ===")
        
        # Collect statistics for new format
        all_locations = []
        all_actions = []
        all_movements = []
        high_confidence_count = 0
        total_analyses = 0
        
        for video_key, result in results.items():
            location_analysis = result.get('location_analysis', [])
            
            # print(f"\nVideo: {video_key}")
            
            if not location_analysis:
                print("  No location analysis available")
                continue
                
            # Display each analysis for this video
            for i, analysis in enumerate(location_analysis):
                primary_location = analysis.get('primary_location', '')
                action_at_location = analysis.get('action_at_location', '')
                user_movement = analysis.get('user_movement', '')
                confidence = analysis.get('confidence', 0.0)
                
                # Collect for statistics
                if primary_location:
                    all_locations.append(primary_location)
                if action_at_location:
                    all_actions.append(action_at_location)
                if user_movement:
                    all_movements.append(user_movement)
                    
                total_analyses += 1
                if confidence >= 0.8:
                    high_confidence_count += 1
                
                # Display analysis
                # if len(location_analysis) > 1:
                #     print(f"  Analysis {i+1}:")
                #     print(f"    Location: {primary_location or 'Not specified'}")
                #     print(f"    Action: {action_at_location or 'Not specified'}")
                #     print(f"    Movement: {user_movement or 'Not specified'}")
                #     print(f"    Confidence: {confidence:.2f}")
                # else:
                #     print(f"  Location: {primary_location or 'Not specified'}")
                #     print(f"  Action: {action_at_location or 'Not specified'}")
                #     print(f"  Movement: {user_movement or 'Not specified'}")
                #     print(f"  Confidence: {confidence:.2f}")
        
        # Overall statistics
        print(f"\n=== Overall Statistics ===")
        print(f"Total analyses: {total_analyses}")
        print(f"High confidence analyses (≥0.8): {high_confidence_count}")
        print(f"High confidence rate: {(high_confidence_count/total_analyses*100):.1f}%" if total_analyses > 0 else "No analyses")
        print(f"Unique locations detected: {len(set(all_locations))}")
        print(f"Unique actions detected: {len(set(all_actions))}")
        print(f"Unique movements detected: {len(set(all_movements))}")
        
        if all_locations:
            location_counts = {}
            for loc in all_locations:
                location_counts[loc] = location_counts.get(loc, 0) + 1
            most_common_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Most common locations: {', '.join([f'{loc}({count})' for loc, count in most_common_locations])}")
            
        if all_movements:
            movement_counts = {}
            for mov in all_movements:
                movement_counts[mov] = movement_counts.get(mov, 0) + 1
            most_common_movements = sorted(movement_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"Most common movements: {', '.join([f'{mov}({count})' for mov, count in most_common_movements])}")


class LocalSegmentDescriptionGenerator(SegmentDescriptionGenerator):
    """
    SegmentDescriptionGenerator that uses litellm to call Ollama directly for local VLM models.
    Only supports single video processing, with prompt_type (structured/descriptive).
    """
    def __init__(self, vlm_model: str, cache_dir: str, ollama_base_url: str = "http://localhost:11434"):
        super().__init__(vlm_model, cache_dir)
        self.ollama_base_url = ollama_base_url

    def process_single_video(self, video_path: str, frames_per_second: float = 1.0, prompt_type: str = "condensed") -> Dict[str, Any]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        frames = self._preprocess_video(video_path, frames_per_second)
        if not frames:
            return {
                "error": "Could not extract frames from video",
                "location_analysis": [{
                    "primary_location": "",
                    "action_at_location": "",
                    "user_movement": "",
                    "confidence": 0.0
                }]
            }
        prompt = self._create_prompt(frames, prompt_type)
        # Prepare messages for litellm
        content = prompt
        messages = [{"role": "user", "content": content}]
        try:
            response = completion(
                model=f"ollama/{self.vlm_model}",
                messages=messages,
                max_tokens=500,
                api_base=self.ollama_base_url
            )
            if prompt_type == "structured":
                response_content = response["choices"][0]["message"]["content"]
                return self._parse_json_response(response_content)
            else:
                return response["choices"][0]["message"]["content"]
        except Exception as e:
            return {
                "error": str(e),
                "location_analysis": [{
                    "primary_location": "",
                    "action_at_location": "",
                    "user_movement": "",
                    "confidence": 0.0
                }]
            }

    def _create_prompt(self, frames: List[str], prompt_type: str = "condensed") -> List[Dict[str, Any]]:
        if prompt_type == "condensed":
            prompt_text = (
                """
                Based on the sequence of images from a kitchen, analyze the person's actions and location. Please provide your answer in a structured JSON format only. No addional descriptions.

                1.  **kitchen_location**: Identify the specific area in few words in the kitchen where the person is primarily located (e.g., 'at the stove', 'by the sink', 'at preparation counter', 'at microwave', 'at fridge', 'at coffee station', 'at dishwasher', 'at oven', 'moving between locations').
                2.  **user_activity**: Describe the main activity in words the person is performing across these frames (e.g., 'chopping vegetables', 'stirring a pot', 'washing dishes', 'moving between locations', and any other activity).
                """
            )
        elif prompt_type == "descriptive":
            prompt_text = (
                """
                Analyze this kitchen activity video sequence and provide a detailed, fine-grained, descriptive analysis of what you observe.

                IMPORTANT INSTRUCTIONS:
                - Do NOT assume or hallucinate actions or objects that are not clearly visible in the frames.
                - Focus ONLY on what the user is actually doing, based on clear visual evidence.
                - This video is part of a larger activity, so we are interested in more fine-grained, step-by-step descriptions, not overly generic summaries.
                - Avoid generic statements like "the person is cooking"; instead, describe specific, observable actions and interactions.

                PRIMARY REQUIREMENTS:

                1. **Precise Location Identification**:
                   - Be equipment/appliance specific: mention specific appliances like coffee machine, kitchen sink, refrigerator, stovetop, microwave, dishwasher.
                   - Avoid generic terms like "kitchen", "counter", "appliance".
                   - If at a counter, specify what's nearby: counter near coffee machine, island prep area.

                2. **Specific Action Description**:
                   - Describe the exact interactions: pressing buttons, turning knobs, opening doors, pulling handles, lifting items, pouring, etc.
                   - Include what objects are being interacted with and how.

                3. **Movement Analysis**:
                   - Describe if the person stays in one location or moves between different areas.
                   - Be specific about destinations and transitions.

                4. **Overall Context**:
                   - Provide context about the kitchen activity sequence, but do not speculate beyond what is visible.
                   - Mention any tools, ingredients, or equipment being used, only if clearly visible.
                   - Describe the flow of actions and any patterns you observe, focusing on observable details.

                Please provide a comprehensive, natural language description that captures all the important, fine-grained details of what you observe in this kitchen activity sequence.
                """
            )
        else:
            prompt_text = super()._create_prompt(frames)[0]  # Use parent's structured prompt
        # Use OpenAI-compatible multimodal format for both prompt types
        content = [{"type": "text", "text": prompt_text}]
        for frame in frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
        return content
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",'-m', type=str, default="gemma3:latest")
    parser.add_argument("--frames_per_second",'-f', type=float, default=1.0)
    parser.add_argument("--prompt_type",'-p', type=str, default="structured")
    args = parser.parse_args()

    video_path = "/Volumes/Phase4 Raw/processed_data/P3/FridgeOpen/P3_069/camera.mp4"

    generator = LocalSegmentDescriptionGenerator(vlm_model=args.model, cache_dir="cache")
    result = generator.process_single_video(video_path, args.frames_per_second, args.prompt_type)
    print(result)

def main():
    """Demonstration of SegmentDescriptionGenerator usage."""
    
    # Configuration
    VLM_MODEL = "gpt-4.1"  # or "gpt-4o" for higher quality
    VIDEO_DIR = "/Users/prasoon/Research/VAX/Datasets/autonomous_phase3/phase3_results/P5data-collection/segments_5.0_0.5/P11-02-20250404_144814_20250404_152429/birdseye"  
    CACHE_DIR = "/Users/prasoon/Research/VAX/Datasets/autonomous_phase3/phase3_results/P5data-collection/descriptions_5.0_0.5/birdseye"     
    PROCESSING_ID = "P11-02-20250404_144814_20250404_152429_birdseye"
    os.makedirs(CACHE_DIR, exist_ok=True)

    
    
    # Example paths for testing
    # VIDEO_DIR = "/Users/prasoon/Research/VAX/test_videos"
    # CACHE_DIR = "/Users/prasoon/Research/VAX/cache/segment_processor"
    
    print("=== Segment Description Generator Demo ===")
    
    # Initialize the processor
    processor = SegmentDescriptionGenerator(
        vlm_model=VLM_MODEL,
        cache_dir=CACHE_DIR,
        batch_size=100  # Adjust based on your needs
    )
    
    # Check if directory exists
    if not os.path.exists(VIDEO_DIR):
        print(f"Video directory not found: {VIDEO_DIR}")
        print("Please update the VIDEO_DIR path in the script")
        return
    
    # Demo option: process a single video synchronously
    print("\n=== Option 1: Single Video Processing (Synchronous) ===")
    
    # Find the first video file for demo
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    if video_files:
        test_video = video_files[0]
        print(f"Processing single video: {os.path.basename(test_video)}")
        
        try:
            result = processor.process_single_video(test_video)
            print("Result:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error processing single video: {e}")
    else:
        print("No MP4 files found for single video demo")
    
    print("\n" + "="*50)
    print("=== Option 2: Batch Processing (Asynchronous) ===")
    
    try:
        # Step 1: Start processing
        print(f"\nStep 1: Processing videos in {VIDEO_DIR}")
        batch_info = processor.process_directory(
            video_dir=VIDEO_DIR,
            file_extensions=['*.mp4'],
            processing_id=PROCESSING_ID
        )
        
        processing_id = batch_info['processing_id']
        print(f"Started processing with ID: {processing_id}")
        
        # Step 2: Wait for results
        print(f"\nStep 2: Waiting for batch processing to complete...")
        print("This may take several minutes depending on the number of videos...")
        
        results, status = processor.get_results(
            processing_id=processing_id,
            max_wait_time=600,  # 10 minutes max
            check_interval=15   # Check every 15 seconds
        )
        
        # Step 3: Display results
        print(f"\nStep 3: Processing completed!")
        print(f"Status: {status['completion_rate']:.1f}% complete")
        print(f"Completed: {len(status['completed'])} batches")
        print(f"Failed: {len(status['failed'])} batches")
        
        if results:
            # Save results to file
            results_file = os.path.join(CACHE_DIR, f"results_{processing_id}.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {results_file}")
            
            # Print summary
            processor.print_results_summary(results)
        else:
            print("No results obtained")
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


def demo_single_video_processing(video_path: str, vlm_model: str = "gpt-4.1"):
    """
    Simple demo function for processing a single video.
    
    Args:
        video_path: Path to the video file
        vlm_model: OpenAI model to use
        
    Returns:
        Analysis result dictionary
    """
    import tempfile
    
    # Create a temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = SegmentDescriptionGenerator(
            vlm_model=vlm_model,
            cache_dir=temp_dir
        )
        
        result = processor.process_single_video(video_path)
        return result


def demo_local_processing(video_path: str, vlm_model: str = "gemma3:latest"):
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = LocalSegmentDescriptionGenerator(
            vlm_model=vlm_model,
            cache_dir=temp_dir
        )
        result = processor.process_single_video(video_path, prompt_type="structured", frames_per_second=1.0)
        return result

if __name__ == "__main__":
    main() 
    