import os
import glob
import json
import argparse
import sys
from datetime import datetime
from typing import List, Dict, Any
from tqdm.auto import tqdm
import dotenv
from litellm import completion

dotenv.load_dotenv()
# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.common.video_processing.SegmentDescriptionGenerator import SegmentDescriptionGenerator, LocalSegmentDescriptionGenerator

# Base directories - adjust these paths as needed
base_results_dir = "/Volumes/Research-Prasoon/OrganicHAR/inhome_evaluation"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate descriptions for video segments using VLM")
    
    # Required arguments
    parser.add_argument("--participant", default="P5data-collection", help="Participant name (e.g., P5data-collection)")
    parser.add_argument("--session_prefix", default="P11", help="Session prefix")
    parser.add_argument("--processing_id_prefix", default="paper_eval2_attempt2", help="Processing ID prefix")
    parser.add_argument("--mode", default="local", choices=["submit", "extract", "sync", "local"], 
                       help="Mode: 'submit' to create/submit batches, 'extract' to get results, 'sync' to process remaining videos synchronously, 'local' to process videos locally one by one")
    
    # Optional arguments
    parser.add_argument("--window_size", default="5.0", help="Window size in seconds (default: 5.0)")
    parser.add_argument("--sliding_window_length", default="0.5", help="Sliding window length (default: 0.5)")
    parser.add_argument("--camera_type", choices=["birdseye", "depth"], default="birdseye",
                       help="Camera type to process (default: birdseye)")
    parser.add_argument("--vlm_model", default="openbmb/minicpm-v2.6", help="VLM model to use (default: gpt-4.1)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing (default: 300)")
    parser.add_argument("--max_wait_time", type=int, default=600, help="Max wait time for results in seconds (default: 600)")
    parser.add_argument("--check_interval", type=int, default=15, help="Check interval for results in seconds (default: 15)")
    
    return parser.parse_args()

def submit_batches(args):
    """Submit video segments for batch processing."""
    print(f"=== Submitting Batches for {args.participant} ===")
    
    # Setup directories
    segments_base_dir = f"{base_results_dir}/{args.participant}/segments_{args.window_size}_{args.sliding_window_length}"
    cache_base_dir = f"{base_results_dir}/{args.participant}/descriptions_{args.window_size}_{args.sliding_window_length}/{args.vlm_model}"
    cache_dir = os.path.join(cache_base_dir, args.camera_type)
    
    print(f"Segments directory: {segments_base_dir}")
    print(f"Cache directory: {cache_dir}")
    
    if not os.path.exists(segments_base_dir):
        print(f"Segments directory not found: {segments_base_dir}")
        return
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initialize processor
    processor = SegmentDescriptionGenerator(
        vlm_model=args.vlm_model,
        cache_dir=cache_dir,
        batch_size=args.batch_size
    )
    
    # Create processing ID
    processing_id = f"{args.participant}_{args.camera_type}_{args.window_size}_{args.sliding_window_length}_{args.vlm_model}_{args.processing_id_prefix}"

    # Load existing results if available
    results_filename = f"results_{args.participant}_{args.camera_type}_{args.window_size}_{args.sliding_window_length}_{args.vlm_model}.json"
    results_file = os.path.join(cache_dir, results_filename)

    if os.path.exists(results_file):
        existing_results = json.load(open(results_file))
        print(f"Loaded {len(existing_results)} existing results")
    else:
        existing_results = {}
    exisiting_video_paths = list(existing_results.keys())
    exisiting_video_paths = [f"{'_'.join(path.split("_")[1:-1])}.mp4" for path in exisiting_video_paths]
    
    print(f"Processing ID: {processing_id}")
    print(f"Looking for {args.camera_type} videos...")
    
    try:
        # Use camera type in file extensions to filter videos
        file_extensions = [f'**/{args.camera_type}/*.mp4']
        
        # Process all videos recursively with camera type filter
        batch_info = processor.process_directory(
            video_dir=segments_base_dir,
            file_extensions=file_extensions,
            processing_id=processing_id,
            exisiting_video_paths=exisiting_video_paths
        )
        
        # Save submission summary with deterministic filename
        submission_summary = {
            'participant': args.participant,
            'camera_type': args.camera_type,
            'window_size': args.window_size,
            'sliding_window_length': args.sliding_window_length,
            'vlm_model': args.vlm_model,
            'processing_id': processing_id,
            'submission_time': datetime.now().isoformat(),
            'batch_ids': batch_info.get('batch_ids', []),
            'total_batches': len(batch_info.get('batch_ids', []))
        }
        
        # Use deterministic filename based on parameters
        summary_filename = f"submission_summary_{args.participant}_{args.camera_type}_{args.window_size}_{args.sliding_window_length}_{args.vlm_model}.json"
        summary_file = os.path.join(cache_dir, summary_filename)
        with open(summary_file, 'w') as f:
            json.dump(submission_summary, f, indent=2)
        
        print(f"\n=== Submission Summary ===")
        print(f"Processing ID: {processing_id}")
        print(f"Total batches submitted: {len(batch_info.get('batch_ids', []))}")
        print(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Error during batch submission: {str(e)}")
        import traceback
        traceback.print_exc()

def extract_results(args):
    """Extract results from submitted batches."""
    print(f"=== Extracting Results for {args.participant} ===")
    
    # Setup directories
    cache_base_dir = f"{base_results_dir}/{args.participant}/descriptions_{args.window_size}_{args.sliding_window_length}/{args.vlm_model}"
    cache_dir = os.path.join(cache_base_dir, args.camera_type)
    
    if not os.path.exists(cache_dir):
        print(f"Cache directory not found: {cache_dir}")
        print("No batches have been submitted yet")
        return
    
    print(f"Cache directory: {cache_dir}")
    
    # Initialize processor
    processor = SegmentDescriptionGenerator(
        vlm_model=args.vlm_model,
        cache_dir=cache_dir,
        batch_size=args.batch_size
    )
    
    # Look for specific submission summary based on current arguments
    summary_filename = f"submission_summary_{args.participant}_{args.camera_type}_{args.window_size}_{args.sliding_window_length}_{args.vlm_model}.json"
    summary_file = os.path.join(cache_dir, summary_filename)
    
    if not os.path.exists(summary_file):
        print(f"Submission summary not found: {summary_filename}")
        print("Make sure to submit batches first with the same parameters")
        return
    
    print(f"Using submission summary: {summary_filename}")
    
    with open(summary_file, 'r') as f:
        submission_summary = json.load(f)
    
    processing_id = submission_summary['processing_id']
    print(f"Processing ID: {processing_id}")
    
    try:
        # Get results for this processing session
        results, status = processor.get_results(
            processing_id=processing_id,
            max_wait_time=args.max_wait_time,
            check_interval=args.check_interval
        )

        # get the results that are in the results dir but not in the results dict
        results_dir = os.path.join(cache_dir, "results")
        results_dir_files = os.listdir(results_dir)
        results_dir_files = [file for file in results_dir_files if file.endswith("_result.json")]
        results_dir_files = [file for file in results_dir_files if file not in results]

        # load the results from the results dir
        for result_file in results_dir_files:
            with open(os.path.join(results_dir, result_file), 'r') as f:
                result = json.load(f)
                result_cache_key = result_file.split("_result.json")[0]
                results[result_cache_key] = result
        
        if results:
            # Save results with deterministic filename
            results_filename = f"results_{args.participant}_{args.camera_type}_{args.window_size}_{args.sliding_window_length}_{args.vlm_model}.json"
            results_file = os.path.join(cache_dir, results_filename)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✓ Extracted {len(results)} results")
            print(f"  Completion rate: {status['completion_rate']:.1f}%")
            print(f"  Completed batches: {len(status['completed'])}")
            print(f"  Failed batches: {len(status['failed'])}")
            print(f"  Results saved to: {results_file}")
            
            # Print summary
            processor.print_results_summary(results)
            
        else:
            print(f"✗ No results available yet")
            print(f"  Status: {status}")
        
        # Save extraction summary with deterministic filename
        extraction_summary = {
            'participant': args.participant,
            'camera_type': args.camera_type,
            'window_size': args.window_size,
            'sliding_window_length': args.sliding_window_length,
            'vlm_model': args.vlm_model,
            'processing_id': processing_id,
            'extraction_time': datetime.now().isoformat(),
            'status': status,
            'results_count': len(results) if results else 0,
            'extraction_success': bool(results)
        }
        
        if results:
            extraction_summary['results_file'] = results_file
        
        extraction_filename = f"extraction_summary_{args.participant}_{args.camera_type}_{args.window_size}_{args.sliding_window_length}_{args.vlm_model}.json"
        extraction_summary_file = os.path.join(cache_dir, extraction_filename)
        with open(extraction_summary_file, 'w') as f:
            json.dump(extraction_summary, f, indent=2)
        
        print(f"\nExtraction summary saved to: {extraction_summary_file}")
        
    except Exception as e:
        print(f"Error extracting results: {str(e)}")
        import traceback
        traceback.print_exc()

def process_requests_sync(args):
    """Process remaining videos synchronously that failed or were missed in batch processing."""
    print(f"=== Processing Remaining Videos Synchronously for {args.participant} ===")
    
    # Setup directories
    segments_base_dir = f"{base_results_dir}/{args.participant}/segments_{args.window_size}_{args.sliding_window_length}"
    cache_base_dir = f"{base_results_dir}/{args.participant}/descriptions_{args.window_size}_{args.sliding_window_length}/{args.vlm_model}"
    cache_dir = os.path.join(cache_base_dir, args.camera_type)
    
    if not os.path.exists(segments_base_dir):
        print(f"Segments directory not found: {segments_base_dir}")
        return
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Segments directory: {segments_base_dir}")
    print(f"Cache directory: {cache_dir}")
    
    # Initialize processor
    processor = SegmentDescriptionGenerator(
        vlm_model=args.vlm_model,
        cache_dir=cache_dir,
        batch_size=args.batch_size
    )
    
    # Find all video files that should be processed
    print(f"Looking for {args.camera_type} videos...")
    file_extensions = [f'**/{args.camera_type}/*.mp4']
    
    video_paths = []
    for ext in file_extensions:
        video_paths.extend(glob.glob(os.path.join(segments_base_dir, ext), recursive=True))
    
    video_paths = sorted(list(set(video_paths)))  # Remove duplicates and sort
    
    # Remove duplicates where the file basename is the same
    video_file_names = [os.path.basename(path) for path in video_paths]
    video_paths = [path for i, path in enumerate(video_paths) if video_file_names[i] not in video_file_names[:i]]
    video_paths = sorted(list(set(video_paths)))
    
    if not video_paths:
        print(f"No video files found in {segments_base_dir}")
        return
    
    print(f"Found {len(video_paths)} total video files")
    
    # Load existing results if available
    results_filename = f"results_{args.participant}_{args.camera_type}_{args.window_size}_{args.sliding_window_length}_{args.vlm_model}.json"
    results_file = os.path.join(cache_dir, results_filename)
    
    existing_results = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        print(f"Loaded {len(existing_results)} existing results")
    else:
        print("No existing results file found, will process all videos")
    
    # Find videos that need processing
    videos_to_process = []
    for video_path in video_paths:
        # Create cache key similar to how SegmentDescriptionGenerator does it
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cache_key = f"activity_{video_name}_1.0".replace('/', '_').replace('\\', '_')
        
        # Check if this video already has results
        if cache_key not in existing_results:
            videos_to_process.append((video_path, cache_key))
    
    if not videos_to_process:
        print("All videos have already been processed!")
        return
    
    print(f"Found {len(videos_to_process)} videos that need processing")
    
    # Process videos synchronously
    new_results = {}
    processed_count = 0
    failed_count = 0
    
    for i, (video_path, cache_key) in tqdm(enumerate(videos_to_process), total=len(videos_to_process), desc="Processing videos"):
        try:
            result = processor.process_single_video(video_path)
            new_results[cache_key] = result
            processed_count += 1
            
        except Exception as e:
            failed_count += 1
            print(f"Error processing {os.path.basename(video_path)}: {str(e)}")
            continue
    
    # Merge new results with existing results
    all_results = {**existing_results, **new_results}
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n=== Synchronous Processing Summary ===")
    print(f"Total videos found: {len(video_paths)}")
    print(f"Previously processed: {len(existing_results)}")
    print(f"Newly processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total results now: {len(all_results)}")
    print(f"Results saved to: {results_file}")
    
    # Save sync summary
    sync_summary = {
        'participant': args.participant,
        'camera_type': args.camera_type,
        'window_size': args.window_size,
        'sliding_window_length': args.sliding_window_length,
        'vlm_model': args.vlm_model,
        'sync_time': datetime.now().isoformat(),
        'total_videos_found': len(video_paths),
        'previously_processed': len(existing_results),
        'newly_processed': processed_count,
        'failed': failed_count,
        'total_results': len(all_results),
        'results_file': results_file
    }
    
    sync_filename = f"sync_summary_{args.participant}_{args.camera_type}_{args.window_size}_{args.sliding_window_length}_{args.vlm_model}.json"
    sync_summary_file = os.path.join(cache_dir, sync_filename)
    with open(sync_summary_file, 'w') as f:
        json.dump(sync_summary, f, indent=2)
    
    print(f"Sync summary saved to: {sync_summary_file}")
    
    # Print results summary for new results only
    if new_results:
        print(f"\n=== New Results Summary ===")
        processor.print_results_summary(new_results)

def process_local(args):
    """Process all videos locally one by one using LocalSegmentDescriptionGenerator, saving after each video."""
    print(f"=== Local Processing for {args.participant} ===")
    
    # Setup directories
    segments_base_dir = f"{base_results_dir}/{args.participant}/segments_{args.window_size}_{args.sliding_window_length}"
    cache_base_dir = f"{base_results_dir}/{args.participant}/descriptions_{args.window_size}_{args.sliding_window_length}/{args.vlm_model.replace("/", "_").replace(":", "_")}"
    cache_dir = os.path.join(cache_base_dir, args.camera_type)
    
    if not os.path.exists(segments_base_dir):
        print(f"Segments directory not found: {segments_base_dir}")
        return
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Segments directory: {segments_base_dir}")
    print(f"Cache directory: {cache_dir}")
    
    # Initialize processor
    processor = LocalSegmentDescriptionGenerator(
        vlm_model=args.vlm_model,
        cache_dir=cache_dir
    )
    
    # Find all video files
    file_extensions = [f'**/{args.camera_type}/*.mp4']
    video_paths = []
    for ext in file_extensions:
        video_paths.extend(glob.glob(os.path.join(segments_base_dir, ext), recursive=True))
    video_paths = sorted(list(set(video_paths)))
    if not video_paths:
        print(f"No video files found in {segments_base_dir}")
        return
    print(f"Found {len(video_paths)} video files")
    
    results_filename = f"results_local_{args.participant}_{args.camera_type}_{args.window_size}_{args.sliding_window_length}_{args.vlm_model.replace("/", "_").replace(":", "_")}.json"
    results_file = os.path.join(cache_dir, results_filename)
    # Load existing results if present
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    for video_path in tqdm(video_paths, desc="Processing videos locally"):
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            cache_key = f"activity_{video_name}_1.0".replace('/', '_').replace('\\', '_')
            if cache_key in results:
                continue  # Skip already processed
            result = processor.process_single_video(video_path, prompt_type="condensed")
            results[cache_key] = result
            # Save after each video
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error processing {os.path.basename(video_path)}: {str(e)}")
            continue
    print(f"\nLocal processing complete. Results saved to: {results_file}")
    print(f"Total videos processed: {len(results)}")

def main():
    """Main function."""
    args = parse_args()
    
    print(f"Starting description generation for participant: {args.participant}")
    print(f"Mode: {args.mode}")
    print(f"Camera type: {args.camera_type}")
    print(f"Window parameters: {args.window_size}s window, {args.sliding_window_length}s step")
    print(f"VLM Model: {args.vlm_model}")
    
    if args.mode == "submit":
        submit_batches(args)
    elif args.mode == "extract":
        extract_results(args)
    elif args.mode == "sync":
        process_requests_sync(args)
    elif args.mode == "local":
        process_local(args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
    
    print("\nCompleted!")
    return 0

if __name__ == "__main__":
    exit(main())
