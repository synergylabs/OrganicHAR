import os
import json
import time
from typing import Dict, Any, List, Optional
import dotenv
from tqdm import tqdm
from google import genai
from google.genai.types import CreateBatchJobConfig
from google.cloud import storage
from pydantic import BaseModel, ValidationError

dotenv.load_dotenv()

class LocationAnalysis(BaseModel):
    primary_location: str
    action_at_location: str
    user_movement: str
    confidence: float

class LocationAnalysisResponse(BaseModel):
    location_analysis: List[LocationAnalysis]

class VertexAIBatchProcessor:
    """
    Vertex AI Batch processor for video segments using the official batch prediction API.
    Splits into push_batch (prepare and upload) and start_batch (submit job) phases.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1", model: str = "gemini-2.5-flash"):
        # Use Vertex AI client instead of API key
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.project_id = project_id
        self.location = location
        self.model = model
        self.storage_client = storage.Client(project=project_id)
        
    def push_batch(self, segments_file: str, processing_id: str, 
                   gcs_bucket: str, output_file: Optional[str] = None, batch_size: int = 100) -> Dict[str, Any]:
        """
        Phase 1: Read videos as bytes and create JSONL batch input file, then upload to GCS.
        
        Args:
            segments_file: Path to file containing video segment paths (one per line).
            processing_id: Custom processing ID for this batch.
            gcs_bucket: GCS bucket name (without gs:// prefix) for storing batch input file.
            output_file: Path to output file to check for existing results (to skip already processed).
            
        Returns:
            Dictionary with upload info and paths needed for start_batch.
        """
        # Read video paths
        with open(segments_file, 'r') as f:
            video_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(video_paths)} video segments to process.")
        
        # Load existing results to skip successful ones
        existing_results = {}
        if output_file and os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_results = json.load(f)
                print(f"Loaded {len(existing_results)} existing results from {output_file}.")
            except Exception as e:
                print(f"Warning: Could not load existing results from {output_file}: {e}.")
        
        batch_requests = []
        skipped_already_processed = 0
        skipped_too_large = 0
        max_video_size_mb = 20  # Gemini limit for inline video data
        
        print("Reading videos as bytes and preparing batch requests...")
        
        for idx, video_path in enumerate(tqdm(video_paths, desc="Processing videos")):
            custom_id = f"{processing_id}_{idx:06d}"
            
            # Check if this segment already has a successful result
            if custom_id in existing_results:
                result = existing_results[custom_id]
                has_error = (
                    "error" in result or 
                    (isinstance(result.get("location_analysis"), list) and 
                     len(result["location_analysis"]) > 0 and 
                     result["location_analysis"][0].get("primary_location") == "" and
                     result["location_analysis"][0].get("confidence", 0) == 0.0)
                )
                
                if not has_error:
                    skipped_already_processed += 1
                    continue
                else:
                    print(f"Re-processing {custom_id} - previous result had error.")
            
            # Check file existence
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}. Skipping.")
                continue
            
            try:
                # Check file size first (same as simple version)
                file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                if file_size_mb > max_video_size_mb:
                    print(f"Skipping {video_path} - file size {file_size_mb:.1f}MB exceeds {max_video_size_mb}MB limit")
                    skipped_too_large += 1
                    continue
                
                # Read video as bytes (same as simple version)
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                
                # Encode video bytes as base64 for JSON serialization
                import base64
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                
                # Create batch request with inline data (similar to simple version)
                batch_request = {
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "text": self._create_prompt_text()
                                    },
                                    {
                                        "inline_data": {
                                            "data": video_base64,
                                            "mime_type": "video/mp4"
                                        }
                                    }
                                ]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.1,
                            "maxOutputTokens": 500,
                            "responseMimeType": "application/json",
                            "responseSchema": {
                                "type": "object",
                                "properties": {
                                    "location_analysis": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "primary_location": {
                                                    "type": "string"
                                                },
                                                "action_at_location": {
                                                    "type": "string"
                                                },
                                                "user_movement": {
                                                    "type": "string"
                                                },
                                                "confidence": {
                                                    "type": "number"
                                                }
                                            },
                                            "required": ["primary_location", "action_at_location", "user_movement", "confidence"]
                                        }
                                    }
                                },
                                "required": ["location_analysis"]
                            }
                        }
                    },
                    # Add custom ID for tracking (this will be preserved in output)
                    "custom_id": custom_id
                }
                
                batch_requests.append(batch_request)
                
                # Check if we've reached the batch size limit
                if len(batch_requests) >= batch_size:
                    break
                
            except Exception as e:
                print(f"Error processing {custom_id} ({video_path}): {e}. Skipping.")
                continue
        
        if skipped_already_processed > 0:
            print(f"Skipped {skipped_already_processed} videos that were already successfully processed.")
        if skipped_too_large > 0:
            print(f"Skipped {skipped_too_large} videos due to size > {max_video_size_mb}MB limit.")
        
        if not batch_requests:
            print("No new segments to process. All are either completed, too large, or had errors.")
            return {"status": "no_new_requests", "batch_requests": 0}
        
        # Create and upload JSONL batch input file to GCS
        batch_input_filename = f"batch_input_{processing_id}.jsonl"
        local_jsonl_path = f"/tmp/{batch_input_filename}"
        
        try:
            # Write JSONL locally first
            with open(local_jsonl_path, 'w') as f:
                for req in batch_requests:
                    f.write(json.dumps(req) + '\n')
            
            # Upload JSONL to GCS
            bucket = self.storage_client.bucket(gcs_bucket)
            gcs_jsonl_path = f"batch_inputs/{batch_input_filename}"
            jsonl_blob = bucket.blob(gcs_jsonl_path)
            jsonl_blob.upload_from_filename(local_jsonl_path)
            
            # Clean up local file
            os.remove(local_jsonl_path)
            
            batch_input_uri = f"gs://{gcs_bucket}/{gcs_jsonl_path}"
            
            print(f"Successfully prepared {len(batch_requests)} requests and uploaded batch file to {batch_input_uri}")
            
            return {
                "status": "ready_for_batch",
                "batch_input_uri": batch_input_uri,
                "batch_requests": len(batch_requests),
                "processing_id": processing_id,
                "gcs_bucket": gcs_bucket
            }
            
        except Exception as e:
            print(f"Error creating batch input file: {e}")
            return {"status": "error", "message": str(e)}
    
    def start_batch(self, push_result: Dict[str, Any], output_gcs_bucket: Optional[str] = None) -> str:
        """
        Phase 2: Start the Vertex AI batch prediction job.
        
        Args:
            push_result: Result from push_batch containing batch_input_uri and other info.
            output_gcs_bucket: GCS bucket for output (defaults to same bucket as input).
            
        Returns:
            Batch job name for tracking.
        """
        if push_result["status"] != "ready_for_batch":
            raise ValueError(f"Cannot start batch: {push_result}")
        
        batch_input_uri = push_result["batch_input_uri"]
        processing_id = push_result["processing_id"]
        
        # Setup output location
        if not output_gcs_bucket:
            output_gcs_bucket = push_result["gcs_bucket"]
        
        output_uri = f"gs://{output_gcs_bucket}/batch_outputs/{processing_id}/"
        
        print(f"Starting Vertex AI batch job...")
        print(f"Input: {batch_input_uri}")
        print(f"Output: {output_uri}")
        print(f"Model: {self.model}")
        
        try:
            # Create batch job using Vertex AI API
            batch_job = self.client.batches.create(
                model=self.model,
                src=batch_input_uri,
                config=CreateBatchJobConfig(dest=output_uri)
            )
            
            print(f"Batch job created: {batch_job.name}")
            print(f"Initial state: {batch_job.state}")
            print(f"You can monitor progress in the Cloud Console:")
            print(f"https://console.cloud.google.com/vertex-ai/batch-predictions")
            
            return batch_job.name
            
        except Exception as e:
            print(f"Error starting batch job: {e}")
            raise

    def get_batch_status(self, batch_job_name: str) -> Dict[str, Any]:
        """Get the current status of a batch job."""
        try:
            batch_job = self.client.batches.get(name=batch_job_name)
            return {
                "name": batch_job.name,
                "state": batch_job.state,
                "create_time": batch_job.create_time,
                "start_time": getattr(batch_job, 'start_time', None),
                "end_time": getattr(batch_job, 'end_time', None),
                "output_uri": getattr(batch_job.dest, 'gcs_uri', None) if hasattr(batch_job, 'dest') else None,
                "error": getattr(batch_job, 'error', None)
            }
        except Exception as e:
            return {"error": f"Failed to get batch status: {e}"}

    def retrieve_results(self, batch_job_name: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve results from a completed Vertex AI batch job.
        
        Args:
            batch_job_name: Name of the batch job to retrieve results from.
            output_file: Optional path to save results to.
            
        Returns:
            Dictionary with results keyed by custom_id.
        """
        try:
            batch_job = self.client.batches.get(name=batch_job_name)
            
            if batch_job.state != "JOB_STATE_SUCCEEDED":
                print(f"Batch job is not in succeeded state. Current state: {batch_job.state}")
                if hasattr(batch_job, 'error') and batch_job.error:
                    print(f"Error: {batch_job.error}")
                return {}
            
            # Get output GCS location
            if not hasattr(batch_job, 'dest') or not batch_job.dest.gcs_uri:
                print("No output URI found in batch job.")
                return {}
            
            output_base_uri = batch_job.dest.gcs_uri
            print(f"Retrieving results from: {output_base_uri}")
            
            # List files in the output directory
            bucket_name = output_base_uri.replace("gs://", "").split("/")[0]
            prefix = "/".join(output_base_uri.replace("gs://", "").split("/")[1:])
            
            bucket = self.storage_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix))
            
            # Find the predictions.jsonl file
            predictions_blob = None
            for blob in blobs:
                if blob.name.endswith("predictions.jsonl"):
                    predictions_blob = blob
                    break
            
            if not predictions_blob:
                print("No predictions.jsonl file found in output.")
                return {}
            
            # Download and parse results
            jsonl_content = predictions_blob.download_as_text()
            results = self._process_vertex_ai_batch_output(jsonl_content)
            
            print(f"Retrieved {len(results)} results from batch job.")
            
            # Save results if output_file specified
            if output_file and results:
                # Load existing results if file exists
                all_results = {}
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r') as f:
                            all_results = json.load(f)
                    except Exception as e:
                        print(f"Warning: Could not load existing results: {e}")
                
                # Merge with new results
                all_results.update(results)
                
                # Save updated results
                try:
                    with open(output_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    print(f"Results saved to: {output_file}")
                    print(f"Total results: {len(all_results)}")
                except Exception as e:
                    print(f"Warning: Could not save results: {e}")
            
            return results
            
        except Exception as e:
            print(f"Error retrieving results: {e}")
            return {}

    def _process_vertex_ai_batch_output(self, jsonl_content: str) -> Dict[str, Any]:
        """
        Parse JSONL output from Vertex AI batch prediction.
        Format is different from the manual batch API.
        """
        results = {}
        for line in jsonl_content.strip().split('\n'):
            if not line:
                continue
            try:
                record = json.loads(line)
                
                # Extract custom_id from the request or use a generated one
                custom_id = record.get("custom_id")
                if not custom_id and "request" in record:
                    # Try to extract from request if available
                    custom_id = f"unknown_{len(results)}"
                
                if "response" in record and record["response"]:
                    # Parse successful response
                    response = record["response"]
                    if "candidates" in response and response["candidates"]:
                        try:
                            # Extract the content from the first candidate
                            candidate = response["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                text_part = candidate["content"]["parts"][0].get("text", "")
                                
                                # Parse JSON response
                                parsed_data = json.loads(text_part)
                                
                                # Validate with Pydantic model
                                structured_response = LocationAnalysisResponse.model_validate(parsed_data)
                                
                                results[custom_id] = {
                                    "location_analysis": [
                                        {
                                            "primary_location": analysis.primary_location,
                                            "action_at_location": analysis.action_at_location,
                                            "user_movement": analysis.user_movement,
                                            "confidence": analysis.confidence
                                        }
                                        for analysis in structured_response.location_analysis
                                    ]
                                }
                            else:
                                results[custom_id] = {
                                    "error": "No content in response",
                                    "location_analysis": [{"primary_location": "", "action_at_location": "", "user_movement": "", "confidence": 0.0}]
                                }
                        except (json.JSONDecodeError, ValidationError) as e:
                            results[custom_id] = {
                                "error": f"Response parsing error: {e}",
                                "location_analysis": [{"primary_location": "", "action_at_location": "", "user_movement": "", "confidence": 0.0}]
                            }
                    else:
                        results[custom_id] = {
                            "error": "No candidates in response",
                            "location_analysis": [{"primary_location": "", "action_at_location": "", "user_movement": "", "confidence": 0.0}]
                        }
                else:
                    # Handle error cases
                    error_msg = record.get("status", "Unknown error")
                    results[custom_id] = {
                        "error": error_msg,
                        "location_analysis": [{"primary_location": "", "action_at_location": "", "user_movement": "", "confidence": 0.0}]
                    }
                    
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}. Line: {line}")
            except Exception as e:
                print(f"Error processing line: {e}. Line: {line}")
        
        return results

    def _create_prompt_text(self) -> str:
        """Create prompt text for Gemini API."""
        return """Analyze this kitchen activity video sequence and provide precise location-based activity analysis.

FOCUS: Generate specific, detailed labels for WHERE actions happen and WHAT actions are performed.

PRIMARY REQUIREMENTS:

1.  **Precise Location Identification**:
    -   Be equipment/appliance specific: "coffee_machine", "kitchen_sink", "refrigerator", "stovetop_burner_1", "microwave", "dishwasher"
    -   NOT generic terms like "kitchen", "counter", "appliance"
    -   If at a counter, specify what's on it: "counter_near_coffee_machine", "island_prep_area"

2.  **Specific Action Description**:
    -   Combine action + object interaction: "pressing_buttons on coffee_machine", "turning_knob on stovetop", "opening_door of refrigerator"
    -   Be precise about the interaction method: "pressing", "turning", "pulling", "lifting", "pouring_into"
    -   Include the target object: "on coffee_machine", "into sink", "from refrigerator"

3.  **Movement Analysis**:
    -   "Stationary" - person staying in same location
    -   "Moving to [specific_location]" - person transitioning between locations
    -   Be specific about destination: "Moving to refrigerator", "Moving to sink_area"

SPECIFICITY GUIDELINES:
-   Equipment-level precision for locations (not room-level)
-   Action-object interaction detail (not just the action)
-   Include directional/spatial information when relevant
-   Always provide at least one location analysis
-   If multiple locations involved, focus on where the primary action occurs

MULTIPLE ANALYSES DECISION:
-   **Single Analysis**: Provide only one when confidence ≥0.8 (clear, definitive observation)
-   **Multiple Analyses**: Provide 2-3 options when confidence <0.8 for any individual interpretation
-   Rank analyses by confidence (highest first)
-   Each analysis should represent a genuinely different interpretation of the scene
-   Don't provide multiple analyses for minor variations (e.g., "left_burner" vs "right_burner")

HANDLING UNCLEAR FRAMES:
-   If frames are blurry, dark, or partially obscured, use your best educated guess
-   Base inferences on visible context clues: kitchen layout, partial equipment visibility, body positioning
-   Use typical kitchen workflows to inform reasonable assumptions
-   When uncertain, provide multiple plausible interpretations with respective confidence scores
-   Examples of scenarios for multiple analyses:
    * Ambiguous appliance interaction → could be "coffee_machine" (0.6) or "microwave" (0.5)
    * Unclear counter activity → could be "food_prep on counter" (0.7) or "reaching_for items_in_cabinet" (0.6)
    * Person near multiple appliances → primary focus could be different locations
-   Always provide your best specific guess rather than generic terms like "unknown_appliance" or "unclear_location"

MAKE SURE THAT YOU DO PROVIDE ATLEAST ONE LOCATION ANALYSIS EVEN IF THE CONFIDENCE IS LOW.
If you are not sure about the location analysis, you can provide a location analysis with confidence 0.0.

The response will be automatically structured with location_analysis containing an array of analysis objects, each with primary_location, action_at_location, user_movement, and confidence fields."""

    def save_batch_info(self, batch_info: Dict[str, Any], processing_id: str, output_dir: str = ".") -> str:
        """Save batch job information for later retrieval."""
        batch_info_file = os.path.join(output_dir, f"{processing_id}_vertex_batch_info.json")
        with open(batch_info_file, 'w') as f:
            json.dump(batch_info, f, indent=2)
        print(f"Batch info saved to: {batch_info_file}")
        return batch_info_file
    
    def load_batch_info(self, batch_info_file: str) -> Dict[str, Any]:
        """Load batch job information from file."""
        if not os.path.exists(batch_info_file):
            raise FileNotFoundError(f"Batch info file not found: {batch_info_file}")
        with open(batch_info_file, 'r') as f:
            return json.load(f)




def main():
    """Example usage of VertexAIBatchProcessor with push_batch/start_batch workflow."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Vertex AI Batch Processing for Video Analysis")
    parser.add_argument("--action", choices=["push", "start", "status", "retrieve", "full"],  default="full",
                       help="Action to perform: push (upload), start (begin job), status (check), retrieve (get results), full (push+start)")
    parser.add_argument("--user", default="P1", help="User ID (e.g., P1)")
    parser.add_argument("--segments_file", help="Path to segments file")
    parser.add_argument("--processing_id", help="Processing ID")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use")
    parser.add_argument("--project_id", default="organichar", help="Google Cloud Project ID")
    parser.add_argument("--location", default="us-central1", help="Google Cloud Location")
    parser.add_argument("--gcs_bucket", default="eval2_batches", help="GCS bucket name (without gs:// prefix)")
    parser.add_argument("--output_file", help="Output file for results")
    parser.add_argument("--batch_job_name", help="Batch job name for status/retrieve")
    parser.add_argument("--batch_info_file", help="File containing batch info from push operation")
    parser.add_argument("--batch_size", default=1000, help="Batch size")
    args = parser.parse_args()
    
    # Set defaults based on user
    if not args.segments_file:
        args.segments_file = f"/Volumes/Vax Storage/phase4_results/ubicomp_results_aug25/{args.user}/segment_files.txt"
    if not args.processing_id:
        args.processing_id = f"{args.user}_multiple_vlm_vertex_batch_v4"
    if not args.output_file:
        args.output_file = f"/Volumes/Vax Storage/phase4_results/ubicomp_results_aug25/{args.user}/multiple_vlm_gemini_vertex_results.json"
    if not args.batch_info_file:
        output_dir = os.path.dirname(args.output_file) if args.output_file else "."
        args.batch_info_file = os.path.join(output_dir, f"{args.processing_id}_vertex_batch_info.json")
    
    # Get required environment variables
    project_id = args.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("Google Cloud Project ID required. Set GOOGLE_CLOUD_PROJECT environment variable or pass --project_id.")
    
    gcs_bucket = args.gcs_bucket or os.getenv("GCS_BUCKET_NAME")
    if not gcs_bucket and args.action in ["push", "start", "full"]:
        raise ValueError("GCS bucket required. Set GCS_BUCKET_NAME environment variable or pass --gcs_bucket.")
    
    processor = VertexAIBatchProcessor(
        project_id=project_id,
        location=args.location,
        model=args.model
    )
    
    if args.action == "push":
        if not os.path.exists(args.segments_file):
            raise FileNotFoundError(f"Segments file not found: {args.segments_file}")
        
        print(f"Phase 1: Pushing batch for user: {args.user}")
        print(f"Processing ID: {args.processing_id}")
        print(f"Input file: {args.segments_file}")
        print(f"GCS bucket: {gcs_bucket}")
        
        push_result = processor.push_batch(
            segments_file=args.segments_file,
            processing_id=args.processing_id,
            gcs_bucket=gcs_bucket,
            output_file=args.output_file,
            batch_size=int(args.batch_size)
        )
        
        if push_result["status"] == "ready_for_batch":
            # Save batch info for start_batch
            batch_info_file = processor.save_batch_info(push_result, args.processing_id, 
                                                       os.path.dirname(args.batch_info_file))
            
            print(f"\n✅ Push phase completed successfully!")
            print(f"Batch requests prepared: {push_result['batch_requests']}")
            print(f"Batch info saved to: {batch_info_file}")
            print(f"\nNext step: Start the batch job with:")
            print(f"python {__file__} start --user {args.user} --batch_info_file {batch_info_file}")
        else:
            print(f"❌ Push phase failed: {push_result.get('message', 'Unknown error')}")
    
    elif args.action == "start":
        if not args.batch_info_file or not os.path.exists(args.batch_info_file):
            raise FileNotFoundError(f"Batch info file not found: {args.batch_info_file}")
        
        print(f"Phase 2: Starting batch job for user: {args.user}")
        print(f"Loading batch info from: {args.batch_info_file}")
        
        push_result = processor.load_batch_info(args.batch_info_file)
        batch_job_name = processor.start_batch(push_result, gcs_bucket)
        
        # Update batch info with job name
        push_result["batch_job_name"] = batch_job_name
        processor.save_batch_info(push_result, args.processing_id, 
                                 os.path.dirname(args.batch_info_file))
        
        print(f"\n✅ Batch job started successfully!")
        print(f"Job name: {batch_job_name}")
        print(f"Monitor at: https://console.cloud.google.com/vertex-ai/batch-predictions")
        print(f"\nCheck status with:")
        print(f"python {__file__} status --batch_job_name {batch_job_name}")
        print(f"\nRetrieve results when complete with:")
        print(f"python {__file__} retrieve --batch_job_name {batch_job_name} --output_file {args.output_file}")
    
    elif args.action == "status":
        if not args.batch_job_name:
            if args.batch_info_file and os.path.exists(args.batch_info_file):
                batch_info = processor.load_batch_info(args.batch_info_file)
                args.batch_job_name = batch_info.get("batch_job_name")
            
            if not args.batch_job_name:
                raise ValueError("Batch job name required. Pass --batch_job_name or --batch_info_file with job name.")
        
        print(f"Checking status for batch job: {args.batch_job_name}")
        status = processor.get_batch_status(args.batch_job_name)
        
        if status['error']:
            print(f"❌ Error getting status: {status['error']}")
        else:
            print(f"Job State: {status['state']}")
            print(f"Created: {status['create_time']}")
            if status.get('start_time'):
                print(f"Started: {status['start_time']}")
            if status.get('end_time'):
                print(f"Ended: {status['end_time']}")
            if status.get('output_uri'):
                print(f"Output URI: {status['output_uri']}")
            if status.get('error'):
                print(f"Error: {status['error']}")
    
    elif args.action == "retrieve":
        if not args.batch_job_name:
            if args.batch_info_file and os.path.exists(args.batch_info_file):
                batch_info = processor.load_batch_info(args.batch_info_file)
                args.batch_job_name = batch_info.get("batch_job_name")
            
            if not args.batch_job_name:
                raise ValueError("Batch job name required. Pass --batch_job_name or --batch_info_file with job name.")
        
        print(f"Retrieving results for batch job: {args.batch_job_name}")
        print(f"Results will be saved to: {args.output_file}")
        
        results = processor.retrieve_results(args.batch_job_name, args.output_file)
        
        if results:
            print(f"\n✅ Results retrieved successfully!")
            print(f"Total results: {len(results)}")
            print(f"Saved to: {args.output_file}")
        else:
            print("❌ No results retrieved. Check job status.")
    
    elif args.action == "full":
        # Full workflow: push + start in one command
        if not os.path.exists(args.segments_file):
            raise FileNotFoundError(f"Segments file not found: {args.segments_file}")
        
        print(f"Full workflow for user: {args.user}")
        print(f"Processing ID: {args.processing_id}")
        
        # Phase 1: Push
        print("\n=== Phase 1: Push Batch ===")
        push_result = processor.push_batch(
            segments_file=args.segments_file,
            processing_id=args.processing_id,
            gcs_bucket=gcs_bucket,
            output_file=args.output_file,
            batch_size=int(args.batch_size)
        )
        
        if push_result["status"] != "ready_for_batch":
            print(f"❌ Push phase failed: {push_result.get('message', 'Unknown error')}")
            return
        
        # Phase 2: Start
        print("\n=== Phase 2: Start Batch ===")
        batch_job_name = processor.start_batch(push_result, gcs_bucket)
        
        # Save batch info
        push_result["batch_job_name"] = batch_job_name
        batch_info_file = processor.save_batch_info(push_result, args.processing_id, 
                                                   os.path.dirname(args.output_file))
        
        print(f"\n✅ Full workflow completed!")
        print(f"Batch requests: {push_result['batch_requests']}")
        print(f"Job name: {batch_job_name}")
        print(f"Batch info: {batch_info_file}")
        print(f"Monitor at: https://console.cloud.google.com/vertex-ai/batch-predictions")
        print(f"\nRetrieve results when complete with:")
        print(f"python {__file__} retrieve --batch_job_name {batch_job_name} --output_file {args.output_file}")

if __name__ == "__main__":
    main()