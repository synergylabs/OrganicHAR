from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
import pickle
import mmwave.dsp as dsp

# throwing sklearn to the problem
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.model_selection import *
from enum import Enum
import os
import glob
import base64
import _pickle
import binascii
import tqdm.auto as tqdm
import cv2

@dataclass
class FeatureResults:
    """Results from feature extraction"""
    features: np.ndarray  # Feature vectors
    feature_names: List[str]  # Names of features
    window_sizes: List[int]  # Size of each window
    window_ids: List[str]  # IDs of windows used

import numpy as np
from typing import List


class MotionType(Enum):
    """Types of motion signatures in Doppler data"""
    STATIONARY = 'stationary'
    SLOW_MOTION = 'slow'
    MEDIUM_MOTION = 'medium'
    FAST_MOTION = 'fast'


@dataclass
class VelocityRange:
    min_vel: float
    max_vel: float

    def contains(self, vel: np.ndarray) -> np.ndarray:
        """
        Check if velocities are within range.

        Args:
            vel: numpy array of velocities

        Returns:
            Boolean array of same shape indicating which velocities are in range
        """
        return (np.abs(vel) >= self.min_vel) & (np.abs(vel) <= self.max_vel)

class DopplerConstants:
    # System specifications
    MAX_VELOCITY = 3.3  # m/s
    MAX_RANGE = 7.94  # meters
    FPS = 4  # frames per second

    # Motion classification ranges (in m/s)
    RANGES = {
        MotionType.STATIONARY: VelocityRange(0.0, 0.2),
        MotionType.SLOW_MOTION: VelocityRange(0.2, 1.0),
        MotionType.MEDIUM_MOTION: VelocityRange(1.0, 2.0),
        MotionType.FAST_MOTION: VelocityRange(2.0, 3.3)
    }

    # Detection thresholds
    MIN_VELOCITY = 0.05  # minimum velocity to consider (m/s)
    RANGE_ZONES = [0, 2.65, 5.3, 7.94]  # Divide range into 3 zones

    # Temporal analysis parameters
    MIN_FRAMES = 4  # Minimum frames required for valid analysis
    TIME_SEGMENTS = 4  # Number of segments to divide sequence into


def create_fixed_windows(session_start_time_s: float, last_time_s: float, window_size_seconds: float, sliding_window_length_seconds: float):
    """
    Create fixed window boundaries for a session.
    Args:
        session_start_time_s: Session start time in seconds
        last_time_s: Last timestamp in seconds
        window_size_seconds: Window size in seconds
        sliding_window_length_seconds: Step size for sliding window in seconds
    Returns:
        List of (win_start, win_end) tuples in seconds
    """
    window_starts = []
    curr_start = session_start_time_s
    while curr_start <= last_time_s:
        window_starts.append(curr_start)
        curr_start += sliding_window_length_seconds
    return [(start, start + window_size_seconds) for start in window_starts]

class DopplerFeatures:
    """Extract features from Doppler motion data windows"""

    def __init__(self):
        self.feature_names = []

    def process_b64_doppler_data(self, raw_data_dir, write_dir, prefix="doppler"):
        """Process raw doppler data from a raw data directory."""
        doppler_files = sorted(glob.glob(f"{raw_data_dir}/{prefix}*.csv"))
        
        all_timestamps = []
        all_raw_data = []
        print(f"Processing {len(doppler_files)} doppler files")
        for doppler_file in doppler_files:
            filename_base = os.path.basename(doppler_file)
            file_data = []
            output_file = os.path.join(write_dir, f"{os.path.basename(doppler_file)}.processed.pb")
            if os.path.exists(output_file):
                processed_data = pickle.load(open(output_file, "rb"))
                for instance_ts, instance_data in tqdm.tqdm(processed_data, desc=f"Loading doppler ({filename_base}) from cache"):
                    all_timestamps.append(instance_ts)
                    all_raw_data.append(instance_data)
                continue

            remainder = ""
            num_chunks = 0
            pbar = tqdm.tqdm(desc=f"Processing doppler ({filename_base}) from raw", total=os.path.getsize(doppler_file))
            with open(doppler_file, "r") as myFile:
                while True:
                    chunk = [remainder]
                    chunk_found = False
                    while not chunk_found:
                        try:
                            line = myFile.readline()
                            if line == "":  # End of File
                                break
                        except:
                            break
                        if " ||" in line:
                            chunk_found = True
                            line, remainder = line.split(" ||")
                        chunk.append(line)
                    chunk = ''.join(chunk)
                    if chunk == "":
                        break
                    if len(chunk.split(" | ")) < 2:
                        print(f"Not enough chunk size, {len(chunk)}, {doppler_file}, already_processed {num_chunks} chunks")
                        break
                    ts, data = self.process_b64_chunk(chunk)
                    if ts == -1:
                        print(f"Error processing chunk: {chunk}, {doppler_file}")
                        break
                    all_timestamps.append(ts)
                    all_raw_data.append(data['det_matrix'])
                    file_data.append((ts, data['det_matrix']))
                    pbar.update(len(chunk))
            # save the file to pickle
            pickle.dump(file_data, open(output_file, "wb"))
            print(f"Processed {doppler_file} with {len(file_data)} chunks")
        return all_timestamps, all_raw_data
    
    def process_b64_chunk(self, chunk):
        """
        Process one line of data from doppler
        :param chunk:
        :return:
        """
        ts, encoded_data = chunk.split(" | ")
        ts = int(ts)
        try:
            data = pickle.loads(base64.b64decode(encoded_data.encode()))
        except _pickle.UnpicklingError:
            # print(f"Found unpickling error in chunk, {chunk}")
            data = dict()
            ts = -1
        except binascii.Error:
            data = dict()
            ts = -1
        assert isinstance(data, dict)

        return (ts, data)

    def generate_training_features(self,
                          windows: List[np.ndarray],
                          window_ids: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Generate features for multiple windows for training purposes specifically.

        Args:
            windows: List of doppler sequences
            window_ids: List of window identifiers

        Returns:
            FeatureResults object
        """
        if len(windows) != len(window_ids):
            raise ValueError("Number of windows and window IDs must match")

        # Extract features for each normalized window
        features = []
        for window in tqdm.tqdm(windows, desc="Extracting doppler features"):
            window_features = self.extract_training_features(window)
            features.append(window_features)

        return features,window_ids

    def extract_features(self, window: np.ndarray, roll_size=20) -> np.ndarray:
        """
        Extract features from a single doppler sequence.

        Args:
            window: Doppler sequence

        Returns:
            Feature vector
        """
        doppler_time_matrix = window
        # get range matrix sliding over time
        rt_matrix = doppler_time_matrix.max(axis=2)
        noise_red_data = dsp.compensation.clutter_removal(rt_matrix)
        noise_red_data[noise_red_data < 0] = 0.
        rt_data = noise_red_data.sum(axis=0).reshape(1, -1)

        # get velocity matrix sliding over time
        vt_matrix_mean = doppler_time_matrix.mean(axis=1)
        vt_matrix_std = doppler_time_matrix.std(axis=1)
        vt_matrix = np.concatenate([vt_matrix_mean, vt_matrix_std], axis=1)
        vt_data = vt_matrix.sum(axis=0).reshape(1, -1)

        # aggregate time data
        doppler_featurized_data = np.concatenate([rt_data, vt_data], axis=1)
        return doppler_featurized_data

    def generate_timestamp_windows(self, timestamps: list, det_matrices: list, windows: list):
        """
        Extract window-based features from Doppler data using provided window boundaries.
        Args:
            timestamps: List of timestamps (ints or floats, in seconds or nanoseconds)
            det_matrices: List of det_matrix arrays (np.ndarray)
            windows: List of (win_start, win_end) tuples in seconds
        Returns:
            Tuple of (window_timestamps, feature_list) - one entry per window. If a window has no frames, None is appended for that window.
        Notes:
            This enables fixed window boundaries across sensors for multimodal alignment.
        """
        if len(det_matrices) == 0 or len(timestamps) == 0 or windows is None:
            return [], []
        # Ensure numpy arrays for easier indexing
        timestamps = np.array(timestamps)
        det_matrices = np.array(det_matrices)
        window_timestamps = []
        feature_list = []
        for win_start, win_end in tqdm.tqdm(windows, desc="Extracting doppler features"):
            indices = np.where((timestamps >= win_start) & (timestamps < win_end))[0]
            if len(indices) == 0:
                continue
            else:
                window = det_matrices[indices]
                window_ts = win_end # use the end of the window as the timestamp for the features
                window_array = np.stack(window, axis=0)
                try:
                    features = self.extract_features(window_array)
                    window_timestamps.append(window_ts)
                    feature_list.append(features.flatten())
                except Exception as e:
                    print(f"Error extracting features for window {win_start}-{win_end}: {e}")
                    continue
        return window_timestamps, np.array(feature_list)
    
    def generate_raw_windows(self, timestamps: list, det_matrices: list, windows: list, session_output_dir: str, selected_windows: list):
        """
        Generate raw windows from doppler data
        """
        # create the session output dir
        os.makedirs(session_output_dir, exist_ok=True)

        # Ensure numpy arrays for easier indexing
        timestamps = np.array(timestamps)
        det_matrices = np.array(det_matrices)
        det_metrics_shape = det_matrices.shape
        window_timestamps = []
        feature_list = []
        for win_start, win_end in tqdm.tqdm(windows, desc="Extracting doppler features"):
            indices = np.where((timestamps >= win_start) & (timestamps < win_end))[0]
            window_timestamps = timestamps[indices]
            if len(indices) == 0:
                continue
            else:
                if win_end in selected_windows:
                    instance_data = [(ts, det_matrix) for ts, det_matrix in zip(window_timestamps, det_matrices[indices])]
                    window_instance_dir = f"{session_output_dir}/{win_end}"
                    os.makedirs(window_instance_dir, exist_ok=True)
                    # save the raw doppler windows as pickle file
                    window_pickle_path = f"{window_instance_dir}/doppler.pkl"
                    with open(window_pickle_path, "wb") as f:
                        pickle.dump(instance_data, f)
                    # save the raw doppler windows as video
                    window_video_path = f"{window_instance_dir}/doppler.mp4"
                    cv2_writer = cv2.VideoWriter(window_video_path, cv2.VideoWriter_fourcc(*'avc1'), 4, (det_metrics_shape[2]*5, (det_metrics_shape[1]*5) - 10))
                    for _, det_matrix in instance_data:
                        cv2_writer.write(self.visualize_doppler(det_matrix))
                    cv2_writer.release()
        return None


    def visualize_doppler(ts, det_matrix):
        """
        Visualize the doppler detection matrix using OpenCV with improved sharpness
        and middle 10 rows removed

        Parameters:
        - ts: Timestamp in nanoseconds
        - det_matrix: Detection matrix from the doppler sensor

        Returns:
        - img_col: Colored image for display with enhanced sharpness
        """
        # Use the original data
        data = det_matrix
        det_matrix_vis = data

        # Convert to float for processing
        img = det_matrix_vis.astype(np.float32)

        # Normalize the image to 0-255 range
        if img.max() != img.min():
            img = 255 * (img - img.min()) / (img.max() - img.min())
        else:
            img = np.zeros_like(img)

        # Convert to uint8 for OpenCV processing
        img = img.astype(np.uint8)

        # Apply color map
        img_col = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)

        # Resize the image by 500% with higher quality interpolation
        img_col = cv2.resize(img_col, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

        # Apply stronger sharpening using a more aggressive kernel
        kernel = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]])
        img_col = cv2.filter2D(img_col, -1, kernel)

        # Apply second-pass sharpening for extra clarity
        img_col = cv2.filter2D(img_col, -1, kernel)

        # Calculate the middle row positions to remove
        h, w = img_col.shape[:2]
        middle_row = h // 2
        rows_to_remove = 10
        start_row = middle_row - rows_to_remove // 2
        end_row = middle_row + rows_to_remove // 2

        # # Create a new image without the middle rows
        top_part = img_col[:start_row, :]
        bottom_part = img_col[end_row:, :]
        img_col = np.vstack((top_part, bottom_part))

        return img_col


if __name__ == "__main__":
    extractor = DopplerFeatures()
    input_session_dir = "/Users/prasoon/Research/VAX/datasets/autonomous_phase3/phase3_processed/P2-data-collection/sessions/P2-28"
    output_session_dir = "/Users/prasoon/Research/VAX/datasets/autonomous_phase3/phase3_results/P2-data-collection/sessions/P2-28/"
    os.makedirs(output_session_dir, exist_ok=True)
    all_timestamps, all_raw_data = extractor.process_b64_doppler_data(input_session_dir, output_session_dir)

    # sort the timestamps and raw data
    sorted_indices = np.argsort(all_timestamps)
    all_timestamps = np.array(all_timestamps)[sorted_indices]
    all_raw_data = np.array(all_raw_data)[sorted_indices]
    
    session_start_time_ns = np.min(all_timestamps)  # Replace with actual session start time in ns
    window_size_seconds = 5.0
    sliding_window_length_seconds = 0.5
    if np.median(all_timestamps) > 1e9:
        all_timestamps = np.array(all_timestamps) / 1e9
    else:
        all_timestamps = np.array(all_timestamps)

    all_raw_data = np.array(all_raw_data)
    last_time = all_timestamps[-1]
    session_start_time_s = session_start_time_ns / 1e9
    windows = create_fixed_windows(session_start_time_s, last_time, window_size_seconds, sliding_window_length_seconds)
    window_timestamps, feature_list = extractor.generate_timestamp_windows(all_timestamps, all_raw_data, windows)
    print(len(window_timestamps), feature_list.shape)


