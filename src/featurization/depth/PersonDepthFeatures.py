from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
from scipy import ndimage
import cv2
import os
import re
from datetime import datetime
import tqdm.auto as tqdm
import traceback
import matplotlib.pyplot as plt
import pickle

try:
    from sensors.camera.core.featurizers.PersonDepthEstimator import PersonDepthEstimator
except:
    from src.sensors.camera.core.featurizers.PersonDepthEstimator import PersonDepthEstimator


# Feature extraction parameters
MOTION_THRESHOLD = 0.1  # Threshold for detecting movement in normalized depth
MIN_PERSON_THRESHOLD = 0.05 # Threshold to consider a pixel as part of a person

# Feature names
FEATURE_NAMES = [
    # Static person features (from overall shape in window)
    'person_area',
    'mean_person_depth',
    'person_shape_continuity',
    'max_person_depth',
    'person_aspect_ratio',
    'person_depth_variance',
    # Motion features (from frame-to-frame changes)
    'active_cells_count',       # where movement occurs
    'movement_magnitude',       # how much depth changes in moving areas
    'centroid_displacement',
    'movement_velocity',
    'direction_changes',
    'vertical_movement_ratio',
    # Distribution features (of the person's shape)
    'radius_of_gyration',
    'spatial_entropy',
]

# Clustering parameters
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 2
CLUSTER_SELECTION_EPSILON = 0.05

# Depth normalization parameters
DEPTH_MIN_PERCENTILE = 1
DEPTH_MAX_PERCENTILE = 99

@dataclass
class FeatureResults:
    """Results from feature extraction"""
    features: np.ndarray
    feature_names: List[str]
    window_ids: List[str]


class PersonDepthFeatures:
    """Feature generation for person-only depth sequences."""

    def __init__(self):
        """Initialize the person depth feature generator."""
        self.feature_names = FEATURE_NAMES
        self.depth_stats = None

    def compute_depth_statistics(self, windows: List[np.ndarray]) -> None:
        """
        Compute depth statistics for normalization from non-zero depth values.
        """
        if not windows:
            raise ValueError("Empty windows list")

        all_depths = []
        for window in windows:
            valid_mask = window > 0
            if np.any(valid_mask):
                all_depths.append(window[valid_mask])
        
        if not all_depths:
            # Handle case where all windows are empty
            self.depth_stats = {'min_depth': 0, 'max_depth': 1, 'depth_range': 1}
            return

        all_depths = np.concatenate(all_depths)
        min_depth = np.percentile(all_depths, DEPTH_MIN_PERCENTILE)
        max_depth = np.percentile(all_depths, DEPTH_MAX_PERCENTILE)

        self.depth_stats = {
            'min_depth': min_depth,
            'max_depth': max_depth,
            'depth_range': max(1e-6, max_depth - min_depth)
        }

    def generate_features(self,
                          windows: List[np.ndarray],
                          window_ids: List[str]) -> FeatureResults:
        """
        Generate features for multiple person-depth windows.
        """
        if len(windows) != len(window_ids):
            raise ValueError("Number of windows and window IDs must match")

        
        features = [self.extract_features(window) for window in windows]

        return FeatureResults(
            features=np.array(features),
            feature_names=self.feature_names,
            window_ids=window_ids
        )

    def extract_features(self, window: np.ndarray, raw_height=None) -> np.ndarray:
        """
        Extract features from a single normalized person-depth window.
        """
        if window.size == 0:
            return np.zeros(len(self.feature_names))
        

        if raw_height is not None:
            # Scale each frame to the raw height while maintaining aspect ratio
            try:
                # Get original frame dimensions (height x width)
                original_height = window.shape[1]  # Frame height
                original_width = window.shape[2]   # Frame width
                
                # Calculate new width maintaining aspect ratio
                raw_width = int(original_width * raw_height / original_height)
                
                # Resize each frame individually using area interpolation
                resized_frames = []
                for frame in window:
                    resized_frame = cv2.resize(frame, (raw_width, raw_height), interpolation=cv2.INTER_AREA)
                    resized_frames.append(resized_frame)
                
                window = np.array(resized_frames)
            except Exception as e:
                raise ValueError(f"Error resizing window to raw height {raw_height}: {str(e)}")

        motion_mask, person_shape_mask = self._detect_person_and_motion(window)
        mean_frame = np.mean(window, axis=0)


        # Static Features
        person_area = np.sum(person_shape_mask)
        mean_person_depth = np.mean(mean_frame[person_shape_mask]) if person_area > 0 else 0
        continuity = self._compute_continuity(person_shape_mask)
        max_depth = np.max(mean_frame[person_shape_mask]) if person_area > 0 else 0
        aspect_ratio = self._compute_aspect_ratio(person_shape_mask)
        person_depth_variance = np.var(mean_frame[person_shape_mask]) if person_area > 0 else 0

        # Motion Features
        active_cells = np.sum(motion_mask)
        
        # Calculate movement magnitude only on the pixels that are actually moving
        if len(window) > 1:
            frame_diffs = np.abs(np.diff(window, axis=0))
            movement_mag = np.mean(frame_diffs[:, motion_mask]) if np.any(motion_mask) else 0.0
        else:
            movement_mag = 0.0

        trajectory = self._compute_centroid_trajectory(window)
        displacement = self._compute_displacement(trajectory)
        velocity = self._compute_velocity(trajectory)
        dir_changes = self._compute_direction_changes(trajectory)
        vert_ratio = self._compute_vertical_ratio(trajectory)
        
        # Distributional Features
        gyration_radius = self._compute_radius_of_gyration(person_shape_mask)
        entropy = self._compute_spatial_entropy(person_shape_mask)

        features = np.array([
            person_area, mean_person_depth, continuity, max_depth, aspect_ratio, person_depth_variance,
            active_cells, movement_mag, displacement, velocity, dir_changes, vert_ratio,
            gyration_radius, entropy
        ])
        
        return np.nan_to_num(features)

    def _detect_person_and_motion(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identifies the person's overall shape and areas of motion within it for a given window.
        """
        if frames.size == 0:
            shape = frames.shape[1:]
            return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)

        # 1. Determine the overall shape of the person across the window.
        # This mask represents all pixels that were part of the person at any point in the window.
        # Using np.max is more robust to movement than np.mean.
        person_shape_mask = np.max(frames, axis=0) > MIN_PERSON_THRESHOLD

        # 2. Detect motion within the frames.
        if len(frames) > 1:
            # Calculate the average absolute difference between consecutive frames.
            frame_diffs = np.abs(np.diff(frames, axis=0))
            avg_frame_diff = np.mean(frame_diffs, axis=0)
            
            # A pixel is considered in motion if the change is above a threshold.
            motion_mask = avg_frame_diff > MOTION_THRESHOLD
            
            # 3. Ensure motion is only considered within the person's shape.
            # This cleans up potential noise at the edges.
            motion_mask &= person_shape_mask
        else:
            # No motion can be calculated from a single frame.
            motion_mask = np.zeros_like(person_shape_mask)

        return motion_mask, person_shape_mask

    def _compute_centroid_trajectory(self, window: np.ndarray) -> np.ndarray:
        """
        Compute centroid trajectory from frames.
        """
        trajectory = []
        for frame in window:
            person_mask = frame > MIN_PERSON_THRESHOLD
            ys, xs = np.where(person_mask)
            if len(xs) > 0:
                trajectory.append([np.mean(ys), np.mean(xs)])
            elif trajectory:
                trajectory.append(trajectory[-1]) # Repeat last known if empty
            else:
                trajectory.append([np.nan, np.nan]) # No person in first frame
        
        trajectory = np.array(trajectory)
        # Interpolate NaNs
        if np.isnan(trajectory).any():
            inds = np.arange(len(trajectory))
            good = ~np.isnan(trajectory[:, 0])
            if np.any(good):
                f_y = np.interp(inds, inds[good], trajectory[good, 0])
                f_x = np.interp(inds, inds[good], trajectory[good, 1])
                trajectory = np.vstack((f_y, f_x)).T
        return trajectory
    
    # Helper methods for feature calculation (mostly unchanged from DepthFeatures)
    def _compute_displacement(self, trajectory: np.ndarray) -> float:
        if len(trajectory) < 2: return 0.0
        return np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)))

    def _compute_velocity(self, trajectory: np.ndarray) -> float:
        if len(trajectory) < 2: return 0.0
        return np.mean(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)))

    def _compute_direction_changes(self, trajectory: np.ndarray, min_movement: float = 0.5) -> int:
        if len(trajectory) < 3: return 0
        vectors = np.diff(trajectory, axis=0)
        magnitudes = np.sqrt(np.sum(vectors**2, axis=1))
        significant_movement = magnitudes > min_movement
        if np.sum(significant_movement) < 2: return 0
        vectors = vectors[significant_movement]
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angle_changes = np.abs(np.diff(angles))
        angle_changes = np.minimum(angle_changes, 2 * np.pi - angle_changes)
        return np.sum(angle_changes > np.pi / 4)

    def _compute_vertical_ratio(self, trajectory: np.ndarray) -> float:
        if len(trajectory) < 2: return 0.0
        movements = np.abs(np.diff(trajectory, axis=0))
        return np.sum(movements[:, 0]) / (np.sum(movements[:, 1]) + 1e-6)

    def _compute_continuity(self, mask: np.ndarray) -> float:
        labeled, num_features = ndimage.label(mask)
        if num_features == 0: return 0.0
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        return np.max(sizes) / (np.sum(mask) + 1e-6)

    def _compute_radius_of_gyration(self, mask: np.ndarray) -> float:
        ys, xs = np.where(mask)
        if len(xs) == 0: return 0.0
        center = np.array([np.mean(ys), np.mean(xs)])
        return np.mean(np.sqrt(np.sum((np.column_stack([ys, xs]) - center)**2, axis=1)))

    def _compute_spatial_entropy(self, mask: np.ndarray) -> float:
        if not np.any(mask): return 0.0
        p = mask.astype(float) / np.sum(mask)
        return -np.sum(p[p > 0] * np.log2(p[p > 0]))

    def _compute_aspect_ratio(self, mask: np.ndarray) -> float:
        ys, xs = np.where(mask)
        if len(xs) == 0: return 1.0
        height = np.max(ys) - np.min(ys) + 1
        width = np.max(xs) - np.min(xs) + 1
        return height / (width + 1e-6)

    def parse_filename_timestamps(self, filename: str):
        """Parse start and end timestamps from filename."""
        basename = os.path.basename(filename)
        pattern = r'.*-(\d{8}_\d{6})_(\d{8}_\d{6})\.mp4$'
        match = re.search(pattern, basename)
        if not match:
            raise ValueError(f"Filename {basename} doesn't match expected pattern.")
        start_time = datetime.strptime(match.groups()[0], '%Y%m%d_%H%M%S')
        end_time = datetime.strptime(match.groups()[1], '%Y%m%d_%H%M%S')
        return start_time, end_time

    def create_fixed_windows(self, session_start_time_s: float, last_time_s: float, window_size_seconds: float, sliding_window_length_seconds: float):
        """Create fixed window boundaries for a session."""
        window_starts = np.arange(session_start_time_s, last_time_s, sliding_window_length_seconds)
        return [(start, start + window_size_seconds) for start in window_starts]

    def generate_timestamp_windows_from_video(self, video_path: str, windows: list,
                                     depth_model_path: str, seg_model_name: str, seg_cache_dir: str,
                                     output_fps: float = 8.0, debug: bool = False, raw_height=None):
        """
        Extracts features from a video file for given time windows using a memory-efficient streaming approach.
        """
        estimator = PersonDepthEstimator(depth_model_path, seg_model_name, seg_cache_dir)
        start_time, end_time = self.parse_filename_timestamps(video_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise ValueError("Could not open video file.")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0 or video_fps <= 0:
            cap.release()
            raise ValueError("Video file has no frames or an invalid FPS.")
            
        frame_skip = max(1, int(video_fps / output_fps))

        # --- Streaming Logic ---
        window_idx = 0
        num_windows = len(windows)
        window_timestamps_out = []
        feature_list = []
        frame_buffer = []
        frame_buffer_times = []
        frame_count = 0

        pbar = tqdm.tqdm(total=total_frames, desc="Streaming features from video")

        while True:
            ret, frame = cap.read()
            if not ret or window_idx >= num_windows:
                break
            
            pbar.update(1)
            
            # Frame skipping logic
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            if frame.shape[1] == 1280:
                frame = frame[:, :640, :]

            frame_time = start_time.timestamp() + (frame_count / video_fps)
            frame_count += 1
            
            # Get person depth map for the current frame
            person_depth, _, _ = estimator.process_frame(frame)
            frame_buffer.append(person_depth)
            frame_buffer_times.append(frame_time)

            if debug:
                viz = estimator.depth_estimator.visualize_depth(person_depth)
                cv2.imshow("Person Depth (Debug)", viz)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            # Process all windows that are now complete
            while window_idx < num_windows and windows[window_idx][1] <= frame_time:
                win_start, win_end = windows[window_idx]
                
                # Select frames from buffer that fall within this window
                indices = [i for i, t in enumerate(frame_buffer_times) if win_start <= t < win_end]
                
                if len(indices) > 0:
                    window_depths = np.array([frame_buffer[i] for i in indices])
                    features = self.extract_features(window_depths, raw_height=raw_height)
                    window_timestamps_out.append(win_end)
                    feature_list.append(features)
                
                window_idx += 1

                # Prune the buffer to remove frames no longer needed
                if window_idx < num_windows:
                    next_win_start = windows[window_idx][0]
                    # Find first index to keep
                    keep_from_idx = next((i for i, t in enumerate(frame_buffer_times) if t >= next_win_start), len(frame_buffer_times))
                    frame_buffer = frame_buffer[keep_from_idx:]
                    frame_buffer_times = frame_buffer_times[keep_from_idx:]
        
        pbar.close()
        cap.release()
        if debug:
            cv2.destroyAllWindows()

        # Process any remaining windows after the video ends
        while window_idx < num_windows:
            win_start, win_end = windows[window_idx]
            indices = [i for i, t in enumerate(frame_buffer_times) if win_start <= t < win_end]
            if len(indices) > 0:
                window_depths = np.array([frame_buffer[i] for i in indices])
                features = self.extract_features(window_depths, raw_height=raw_height)
                window_timestamps_out.append(win_end)
                feature_list.append(features)
            window_idx += 1

        # E release the estimator
        del estimator
            
        return window_timestamps_out, feature_list
    
    def generate_raw_windows(self, video_path: str, windows: list, session_output_dir: str, selected_windows: list, camera_name: str,
                                depth_model_path: str, seg_model_name: str, seg_cache_dir: str,
                                output_fps: float = 8.0, debug: bool = False, raw_height=None):
        """
        Extracts features from a video file for given time windows using a memory-efficient streaming approach.
        """
        # create the session output dir
        os.makedirs(session_output_dir, exist_ok=True)

        estimator = PersonDepthEstimator(depth_model_path, seg_model_name, seg_cache_dir)
        start_time, end_time = self.parse_filename_timestamps(video_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise ValueError("Could not open video file.")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0 or video_fps <= 0:
            cap.release()
            raise ValueError("Video file has no frames or an invalid FPS.")
            
        frame_skip = max(1, int(video_fps / output_fps))

        # --- Streaming Logic ---
        window_idx = 0
        num_windows = len(windows)
        frame_buffer = []
        frame_buffer_times = []
        frame_count = 0

        pbar = tqdm.tqdm(total=total_frames, desc="Streaming raw depth from video")

        while True:
            ret, frame = cap.read()
            if not ret or window_idx >= num_windows:
                break
            
            pbar.update(1)
            
            # Frame skipping logic
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            if frame.shape[1] == 1280:
                frame = frame[:, :640, :]

            frame_time = start_time.timestamp() + (frame_count / video_fps)
            frame_count += 1
            
            # Store raw frame for later processing
            frame_buffer.append(frame.copy())
            frame_buffer_times.append(frame_time)

            # Process all windows that are now complete
            while window_idx < num_windows and windows[window_idx][1] <= frame_time:
                win_start, win_end = windows[window_idx]
                
                if win_end in selected_windows:
                    # Select frames from buffer that fall within this window
                    indices = [i for i, t in enumerate(frame_buffer_times) if win_start <= t < win_end]
                    window_frame_timestamps = [frame_buffer_times[i] for i in indices]
                    
                    # Process frames through depth estimation only for selected windows
                    window_depths = []
                    window_depths_viz = []
                    for i in indices:
                        raw_frame = frame_buffer[i]
                        person_depth, person_depth_viz, _ = estimator.process_frame(raw_frame)
                        window_depths.append(person_depth)
                        window_depths_viz.append(person_depth_viz)
                        
                        if debug:
                            viz = estimator.depth_estimator.visualize_depth(person_depth)
                            cv2.imshow("Person Depth (Debug)", viz)
                            if cv2.waitKey(1) & 0xFF == ord('q'): break
                    
                    instance_data = [(ts, depth) for ts, depth in zip(window_frame_timestamps, window_depths)]
                    if len(instance_data) > 0:
                        window_instance_dir = f"{session_output_dir}/{win_end}"
                        os.makedirs(window_instance_dir, exist_ok=True)
                        # save the raw depth windows as pickle file
                        window_pickle_path = f"{window_instance_dir}/persondepth_{camera_name}.pkl"
                        with open(window_pickle_path, "wb") as f:
                            pickle.dump(instance_data, f)
                        # save the raw depth windows as video
                        window_video_path = f"{window_instance_dir}/persondepth_{camera_name}.mp4"
                        window_depths_viz = np.array(window_depths_viz)
                        cv2_writer = cv2.VideoWriter(window_video_path, cv2.VideoWriter_fourcc(*'avc1'), output_fps, (window_depths_viz.shape[2], window_depths_viz.shape[1]))
                        for frame_viz in window_depths_viz:
                            cv2_writer.write(frame_viz)
                        cv2_writer.release()
                
                window_idx += 1
                # Prune the buffer to remove frames no longer needed
                if window_idx < num_windows:
                    next_win_start = windows[window_idx][0]
                    # Find first index to keep
                    keep_from_idx = next((i for i, t in enumerate(frame_buffer_times) if t >= next_win_start), len(frame_buffer_times))
                    frame_buffer = frame_buffer[keep_from_idx:]
                    frame_buffer_times = frame_buffer_times[keep_from_idx:]
        
        pbar.close()
        cap.release()
        if debug:
            cv2.destroyAllWindows()

        # Process any remaining windows after the video ends
        while window_idx < num_windows:
            win_start, win_end = windows[window_idx]
            if win_end in selected_windows:
                indices = [i for i, t in enumerate(frame_buffer_times) if win_start <= t < win_end]
                if len(indices) > 0:
                    window_frame_timestamps = [frame_buffer_times[i] for i in indices]
                    
                    # Process frames through depth estimation
                    window_depths = []
                    window_depths_viz = []
                    for i in indices:
                        raw_frame = frame_buffer[i]
                        person_depth, person_depth_viz, _ = estimator.process_frame(raw_frame)
                        window_depths.append(person_depth)
                        window_depths_viz.append(person_depth_viz)
                    
                    instance_data = [(ts, depth) for ts, depth in zip(window_frame_timestamps, window_depths)]
                    window_instance_dir = f"{session_output_dir}/{win_end}"
                    os.makedirs(window_instance_dir, exist_ok=True)
                    # save the raw depth windows as pickle file
                    window_pickle_path = f"{window_instance_dir}/persondepth_{camera_name}.pkl"
                    with open(window_pickle_path, "wb") as f:
                        pickle.dump(instance_data, f)
                    # save the raw depth windows as video
                    window_video_path = f"{window_instance_dir}/persondepth_{camera_name}.mp4"
                    window_depths_viz = np.array(window_depths_viz)
                    cv2_writer = cv2.VideoWriter(window_video_path, cv2.VideoWriter_fourcc(*'avc1'), output_fps, (window_depths_viz.shape[2], window_depths_viz.shape[1]))
                    for frame_viz in window_depths_viz:
                        cv2_writer.write(frame_viz)
                    cv2_writer.release()
            window_idx += 1

        # E release the estimator
        del estimator
        return None

