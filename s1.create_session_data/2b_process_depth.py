import os
import re
import cv2
import datetime
import numpy as np
from typing import Dict
import subprocess
import ffmpeg
import argparse
import json

def convert_to_mp4(mkv_file, out_dir=None):
    name, ext = os.path.splitext(mkv_file)
    out_name = name + ".mp4"
    if out_dir:
        out_name = os.path.join(out_dir, os.path.basename(out_name))
    if os.path.exists(out_name):
        print("File already exists: {}".format(out_name))
        return True
    try:
        ffmpeg.input(mkv_file).output(out_name).run(overwrite_output=True)
        print("Finished converting {}".format(mkv_file))
        return True
    except ffmpeg.Error as e:
        print("Error converting {}: {}".format(mkv_file, e.stderr))
        return False


def get_video_metadata(video_path):
    """
    Get video metadata (fps, duration, frame count) using multiple methods for reliability

    Args:
        video_path: Path to video file

    Returns:
        tuple: (fps, duration, frame_count, width, height)
    """
    # Try OpenCV first
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    print(f"\nSuccessfully Opened video file: {video_path}\n")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    opencv_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Try to get duration using ffprobe (more reliable)
    try:
        result = subprocess.run([
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ], capture_output=True, text=True, check=True)

        duration = float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        # Fallback to estimating duration from OpenCV if ffprobe fails
        if opencv_fps > 0 and frame_count > 0:
            duration = frame_count / opencv_fps
        else:
            # Last resort default
            duration = 0
            print(f"Warning: Could not determine duration for {video_path}")

    # Determine FPS
    if opencv_fps > 0:
        # Use OpenCV FPS if available
        fps = opencv_fps
    elif duration > 0 and frame_count > 0:
        # Calculate FPS from duration and frame count
        fps = frame_count / duration
    else:
        # Default FPS if all else fails
        fps = 10.0
        print(f"Warning: Could not determine FPS for {video_path}, using default {fps}")

    # Fail-safe for invalid metadata
    if width <= 0 or height <= 0:
        print(f"Warning: Invalid dimensions for {video_path}, using defaults")
        width, height = 1280, 720

    if frame_count <= 0 and duration > 0 and fps > 0:
        # Estimate frame count from duration and FPS
        frame_count = int(duration * fps)

    return (fps, duration, frame_count, width, height)


def add_timestamp(frame, timestamp_str):
    """
    Add timestamp to the bottom left of the frame

    Args:
        frame: Video frame
        timestamp_str: Timestamp string to add
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White
    thickness = 1
    text_size = cv2.getTextSize(timestamp_str, font, font_scale, thickness)[0]

    # Position for bottom left with padding
    position = (10, frame.shape[0] - 10)

    # Add black background for better readability
    bg_top_left = (position[0] - 2, position[1] - text_size[1] - 2)
    bg_bottom_right = (position[0] + text_size[0] + 2, position[1] + 2)
    cv2.rectangle(frame, bg_top_left, bg_bottom_right, (0, 0, 0), -1)

    # Add text
    cv2.putText(frame, timestamp_str, position, font, font_scale, font_color, thickness, cv2.LINE_AA)


def process_video_with_timestamps(input_path, output_path, start_time, fps=None, trim_start=0, trim_end=None):
    """
    Process video file: flip vertically and add timestamps

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        start_time: Datetime object of the video start time
        fps: Frames per second (if None, will be detected from video)
        trim_start: Start time in seconds to trim from beginning
        trim_end: End time in seconds (from start) to trim to
    """
    # Get video metadata
    video_fps, duration, frame_count, width, height = get_video_metadata(input_path)

    # Use provided fps if valid, otherwise use the one from the video
    if fps is None or fps <= 0:
        fps = video_fps

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    # Create output video
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video file: {output_path}")

    # Calculate start and end frames
    start_frame = int(trim_start * fps) if trim_start > 0 else 0
    end_frame = int(trim_end * fps) if trim_end is not None else frame_count

    # Ensure end_frame is valid
    if end_frame <= 0 or end_frame > frame_count:
        end_frame = frame_count

    # Set starting frame position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Process each frame
    current_frame = start_frame
    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # # Rotate the frame 180 degrees (correct upside-down)
        # frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Calculate timestamp for current frame
        frame_time = start_time + datetime.timedelta(seconds=(current_frame / fps))
        timestamp_str = frame_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Truncate microseconds to milliseconds

        # Add timestamp
        add_timestamp(frame, timestamp_str)

        # Write frame
        out.write(frame)
        current_frame += 1

    # Release resources
    cap.release()
    out.release()


def create_blank_video_with_timestamps(output_path, width, height, fps, start_time, duration_seconds):
    """
    Create a blank video with timestamps

    Args:
        output_path: Path to output video file
        width: Video width
        height: Video height
        fps: Frames per second
        start_time: Datetime object of the start time
        duration_seconds: Duration of the video in seconds
    """
    # Create blank frame
    blank_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Create video writer
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    if not out.isOpened():
        raise ValueError(f"Could not create output video file: {output_path}")

    # Calculate number of frames
    frame_count = int(duration_seconds * fps)

    # Generate frames with timestamps
    for i in range(frame_count):
        # Calculate timestamp for current frame
        frame_time = start_time + datetime.timedelta(seconds=i / fps)
        timestamp_str = frame_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Truncate microseconds to milliseconds

        # Add timestamp to the frame
        frame_with_timestamp = blank_frame.copy()
        add_timestamp(frame_with_timestamp, timestamp_str)

        # Write frame
        out.write(frame_with_timestamp)

    # Release resources
    out.release()


def get_mp4_path(file_path: str, mp4_dir: str) -> str:
    """
    Convert file to mp4 if needed and return the mp4 path.
    
    Args:
        file_path: Path to the original file
        mp4_dir: Directory for converted mp4 files
        
    Returns:
        str: Path to the mp4 file
    """
    if file_path.endswith('.mp4'):
        # If already mp4, move to mp4 directory if not already there
        mp4_path = os.path.join(mp4_dir, os.path.basename(file_path))
        if not os.path.exists(mp4_path):
            if os.path.exists(file_path):
                os.rename(file_path, mp4_path)
        return mp4_path
    elif file_path.endswith('.mkv'):
        # Convert mkv to mp4
        mp4_filename = os.path.splitext(os.path.basename(file_path))[0] + '.mp4'
        mp4_path = os.path.join(mp4_dir, mp4_filename)
        
        # Only convert if mp4 doesn't already exist
        if not os.path.exists(mp4_path):
            success = convert_to_mp4(file_path, mp4_dir)
            if not success:
                raise ValueError(f"Failed to convert {file_path} to mp4")
        
        return mp4_path
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def format_video_files(folder_path: str, sessions: Dict[str, Dict[str, str]], output_dir: str) -> Dict[str, str]:
    """
    Format video files based on session start and end times.

    Args:
        folder_path (str): Path to folder containing video files
        sessions (Dict[str, Dict[str, str]]): Dictionary with session names as keys and
                                             start/end timestamps as values
                                             Format: {"session_name": {"start": "YYYYMMDD_HHMMSS",
                                                                      "end": "YYYYMMDD_HHMMSS"}}
        output_dir: str: Path to output directory for processed videos
    Returns:
        Dict[str, str]: Dictionary with session names as keys and paths to processed videos as values
    """
    # Get all video files in the folder (both mp4 and mkv)
    all_video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.mkv'))]

    # Extract timestamps from filenames and sort files
    file_timestamps = {}
    timestamp_pattern = r'(\d{8}_\d{6})\.(mp4|mkv)$'

    for file in all_video_files:
        match = re.search(timestamp_pattern, file)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            file_timestamps[file] = timestamp

    # Sort files by timestamp
    sorted_files = sorted(file_timestamps.keys(), key=lambda x: file_timestamps[x])

    if not sorted_files:
        print("No video files found with valid timestamps")
        return {}

    # Create mp4 conversion directory for on-demand conversion
    mp4_file_dir = os.path.join(folder_path, "mp4_converted")
    os.makedirs(mp4_file_dir, exist_ok=True)

    # Process each session
    result = {}
    for session_name, session_times in sessions.items():
        print(f"Processing session: {session_name}")
        start_time = datetime.datetime.strptime(session_times["start"], "%Y%m%d_%H%M%S")
        end_time = datetime.datetime.strptime(session_times["end"], "%Y%m%d_%H%M%S")

        # Find video files within the session time range
        session_files = []
        for file in sorted_files:
            file_time = file_timestamps[file]
            original_video_path = os.path.join(folder_path, file)
            
            try:
                # Convert to mp4 on-demand only for files that might be in session range
                mp4_video_path = get_mp4_path(original_video_path, mp4_file_dir)
                
                # Get video metadata from the mp4 file
                fps, duration, _, _, _ = get_video_metadata(mp4_video_path)

                # Calculate end time
                file_end_time = file_time + datetime.timedelta(seconds=duration)

                # Check if the file overlaps with the session time range
                if file_end_time > start_time and file_time < end_time:
                    session_files.append((os.path.basename(mp4_video_path), file_time, file_end_time, fps))
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue

        if not session_files:
            print(f"No files found for session {session_name}")
            continue

        os.makedirs(output_dir, exist_ok=True)

        # Get video properties from the first valid video
        first_video = os.path.join(mp4_file_dir, session_files[0][0])
        try:
            _, _, _, width, height = get_video_metadata(first_video)
        except Exception as e:
            print(f"Error getting metadata for first video: {str(e)}")
            continue

        # Use the most common fps from all videos in the session
        all_fps = [fps for _, _, _, fps in session_files]
        if not all_fps:
            print(f"Error: No valid FPS values found for session {session_name}")
            continue

        # Use the most common fps as our session fps (mode)
        from collections import Counter
        session_fps = Counter(all_fps).most_common(1)[0][0]

        print(f"Using session FPS: {session_fps}")

        # Create temporary file list for ffmpeg concatenation
        temp_files = []

        # Process each file in the session
        current_time = start_time
        for i, (file, file_start, file_end, _) in enumerate(session_files):
            file_path = os.path.join(mp4_file_dir, file)

            # If there's a gap between current time and file start, fill with blank frames
            if file_start > current_time:
                gap_seconds = (file_start - current_time).total_seconds()
                if gap_seconds > 0.1:  # Only create gap if it's significant (> 100ms)
                    print(f"Creating {gap_seconds:.2f}s gap video before {file}")
                    blank_video_path = os.path.join(output_dir, f"{session_name}_gap_{i}.mp4")

                    try:
                        # Create blank video with timestamps
                        create_blank_video_with_timestamps(
                            blank_video_path,
                            width,
                            height,
                            session_fps,
                            current_time,
                            gap_seconds
                        )
                        temp_files.append(blank_video_path)
                    except Exception as e:
                        print(f"Error creating gap video: {str(e)}")

            # Process video file
            clip_path = os.path.join(output_dir, f"{session_name}_clip_{i}.mp4")

            # Calculate start and end trim points if needed
            trim_start = 0
            if file_start < start_time:
                trim_start = (start_time - file_start).total_seconds()

            trim_end = None
            if file_end > end_time:
                trim_end = (end_time - file_start).total_seconds()

            print(
                f"Processing video {file} (trim_start={trim_start:.2f}s, trim_end={trim_end if trim_end else None}s)")

            try:
                # Process the video frame by frame to flip and add timestamp
                process_video_with_timestamps(
                    file_path,
                    clip_path,
                    file_start,
                    session_fps,
                    trim_start,
                    trim_end
                )
                temp_files.append(clip_path)
                current_time = min(file_end, end_time)
            except Exception as e:
                print(f"Error processing video {file}: {str(e)}")
                # Continue with next file if this one fails
                continue

        # If session ends after last file, add final blank segment
        if end_time > current_time:
            gap_seconds = (end_time - current_time).total_seconds()
            if gap_seconds > 0.1:  # Only create gap if it's significant (> 100ms)
                print(f"Creating {gap_seconds:.2f}s final gap video")
                blank_video_path = os.path.join(output_dir, f"{session_name}_gap_final.mp4")

                try:
                    # Create blank video with timestamps
                    create_blank_video_with_timestamps(
                        blank_video_path,
                        width,
                        height,
                        session_fps,
                        current_time,
                        gap_seconds
                    )
                    temp_files.append(blank_video_path)
                except Exception as e:
                    print(f"Error creating final gap video: {str(e)}")

        # Skip concatenation if no valid clips were created
        if not temp_files:
            print(f"No valid clips were created for session {session_name}")
            continue

        # Create final video - if only one clip, just rename it
        if len(temp_files) == 1:
            output_path = os.path.join(output_dir, f"{session_name}.mp4")
            os.rename(temp_files[0], output_path)
            print(f"Created session video: {output_path}")
            result[session_name] = output_path
        else:
            # Create file list for ffmpeg concatenation
            concat_file = os.path.join(output_dir, f"{session_name}_concat_list.txt")
            with open(concat_file, 'w') as f:
                for temp_file in temp_files:
                    f.write(f"file '{os.path.abspath(temp_file)}'\n")

            # Concatenate all clips to create final session video
            output_path = os.path.join(output_dir, f"{session_name}.mp4")
            try:
                # Use ffmpeg with copy codec for fast concatenation
                subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
                    "-c", "copy", output_path
                ], check=True)

                print(f"Created session video: {output_path}")
                result[session_name] = output_path
            except subprocess.SubprocessError as e:
                print(f"Error concatenating videos for session {session_name}: {str(e)}")

                # Fallback to frame-by-frame concatenation if ffmpeg concat fails
                try:
                    print("Attempting frame-by-frame concatenation...")

                    # Create a video writer for the output
                    final_out = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        session_fps,
                        (width, height)
                    )

                    # Read and write each frame from each temp file
                    for temp_file in temp_files:
                        temp_cap = cv2.VideoCapture(temp_file)
                        while temp_cap.isOpened():
                            ret, frame = temp_cap.read()
                            if not ret:
                                break
                            final_out.write(frame)
                        temp_cap.release()

                    final_out.release()
                    print(f"Successfully created session video through frame-by-frame concatenation: {output_path}")
                    result[session_name] = output_path
                except Exception as e2:
                    print(f"Frame-by-frame concatenation also failed: {str(e2)}")

            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

            try:
                os.remove(concat_file)
            except OSError:
                pass

    return result


# Example usage:
if __name__ == "__main__":
    base_raw_dir = os.environ.get("BASE_RAW_DIR", "/Volumes/Autonomous2/EHP")
    base_processed_dir = os.environ.get("BASE_PROCESSED_DIR", "./processed_data")

    parser = argparse.ArgumentParser(description="Process video sessions.")
    parser.add_argument("--participant", default="prasoon-data-collection", help="Participant name")
    parser.add_argument("--session_prefix", default="P1", help="Session prefix")
    args = parser.parse_args()

    participant = args.participant
    session_prefix = args.session_prefix

    participant_raw_dir = os.path.join(base_raw_dir, participant)
    participant_processed_dir = os.path.join(base_processed_dir, participant)

    # get the filtered sessions from watch
    filtered_session_file = os.path.join(participant_processed_dir, "watch_sessions_filtered.json")
    filtered_sessions = json.loads(open(filtered_session_file, "r").read())


    depth_folder = f"{base_raw_dir}/{participant}/vision/depth/"
    sessions = {
        f"{session_prefix}-{str(session_idx).zfill(2)}-{session_data['start']}_{session_data['end']}": {
            'start': session_data['start'],
            'end': session_data['end']
        }
        for session_idx, session_data in enumerate(filtered_sessions)
    }
    output_dir = os.path.join(participant_processed_dir, "processed_depth_data")

    processed_videos = format_video_files(depth_folder, sessions, output_dir)
    print(f"Processed videos: {processed_videos}")