# pose_features.py

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from scipy.signal import savgol_filter
from numpy.lib.stride_tricks import sliding_window_view
import pickle
from enum import Enum
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
try:
    from sensors.camera.core.featurizers.YoloPoseEstimator import YoloPoseEstimator
except:
    #from the current directory
    from src.sensors.camera.core.featurizers.YoloPoseEstimator import YoloPoseEstimator
    
from datetime import datetime, timedelta
import tqdm.auto as tqdm



@dataclass
class FeatureResults:
    """Results from feature extraction"""
    features: np.ndarray  # Feature vectors
    feature_names: List[str]  # Names of features


class PoseSignature(Enum):
    """Types of pose signatures"""
    STANDING = 'standing'
    REACHING = 'reaching'
    BENDING = 'bending'
    MANIPULATING = 'manipulating'


@dataclass
class JointPair:
    """Represents a pair of joints for relative measurements"""
    joint1_idx: int
    joint2_idx: int

    def get_distance(self, pose: np.ndarray) -> float:
        """Calculate distance between joints in a pose"""
        j1 = pose[self.joint1_idx]
        j2 = pose[self.joint2_idx]
        return np.sqrt(np.sum((j1[:2] - j2[:2]) ** 2))


class PoseConstants:
    # System specifications
    FRAME_RATE = 8  # fps
    WINDOW_DURATION = 5  # seconds
    MIN_CONFIDENCE = 0.3  # Minimum confidence for valid joint detection

    # YOLO/COCO Landmark indices (17 keypoints)
    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    L_EAR = 3
    R_EAR = 4
    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12
    L_KNEE = 13
    R_KNEE = 14
    L_ANKLE = 15
    R_ANKLE = 16

    # Important joint pairs for feature extraction (updated for YOLO/COCO)
    LIMB_PAIRS = [
        JointPair(L_SHOULDER, L_ELBOW),  # Left upper arm
        JointPair(L_ELBOW, L_WRIST),     # Left forearm
        JointPair(R_SHOULDER, R_ELBOW),  # Right upper arm
        JointPair(R_ELBOW, R_WRIST),     # Right forearm
        JointPair(L_SHOULDER, L_HIP),    # Left torso
        JointPair(R_SHOULDER, R_HIP),    # Right torso
    ]

    # Analysis parameters
    MIN_FRAMES = 10  # Minimum frames required for valid analysis
    TIME_SEGMENTS = 4  # Number of segments for temporal analysis

    # Working height zones (relative to person height)
    # 0.0 = head level, 1.0 = hip level, >1.0 = below hip
    ZONE_THRESHOLDS = {
        'high': (0.0, 0.3),    # Above shoulder level (overhead reach)
        'mid': (0.3, 0.8),     # Shoulder to mid-torso level (normal work)
        'low': (0.8, 1.5)      # Mid-torso to below hip (below counter)
    }

    # Valid range for spatial coordinates
    MIN_COORD = 0
    MAX_COORD = 500  # Assuming max image dimension

    # Clustering parameters
    MIN_CLUSTER_SIZE = 4
    MAX_NOISE_RATIO = 0.3
    TARGET_NUM_CLUSTERS = (20, 40)  # Min, max desired clusters


class PoseFeatures:
    def __init__(self):
        self.feature_names = []
        self._setup_feature_names()

    def _setup_feature_names(self):
        """Initialize list of feature names"""
        self.feature_names = ['hands_mean_velocity', 'hands_max_velocity', 'hands_velocity_std',
                              'hands_direction_changes', 'elbows_mean_velocity', 'elbows_max_velocity',
                              'elbows_velocity_std', 'elbows_direction_changes', 'head_mean_velocity',
                              'head_max_velocity', 'head_velocity_std', 'head_direction_changes',
                              'shoulders_mean_velocity', 'shoulders_max_velocity', 'shoulders_velocity_std',
                              'shoulders_direction_changes', 'hands_mean_acceleration', 'hands_max_acceleration',
                              'hands_acceleration_std', 'elbows_mean_acceleration', 'elbows_max_acceleration',
                              'elbows_acceleration_std', 'head_mean_acceleration', 'head_max_acceleration',
                              'head_acceleration_std', 'mean_orientation', 'orientation_range', 'total_rotation',
                              'max_rotation', 'dynamic_ratio', 'longest_static_duration', 'longest_dynamic_duration',
                              'state_transitions', 'com_vertical_range', 'com_horizontal_range', 'com_path_length',
                              'com_mean_height', 'com_std_horizontal', 'right_hand_max_reach', 'right_hand_mean_reach',
                              'right_hand_vertical_range', 'right_hand_horizontal_range', 'right_hand_mean_height',
                              'right_hand_workspace_area', 'left_hand_max_reach', 'left_hand_mean_reach',
                              'left_hand_vertical_range', 'left_hand_horizontal_range', 'left_hand_mean_height',
                              'left_hand_workspace_area', 'right_upper_arm_mean', 'right_upper_arm_std',
                              'right_upper_arm_range', 'right_forearm_mean', 'right_forearm_std', 'right_forearm_range',
                              'left_upper_arm_mean', 'left_upper_arm_std', 'left_upper_arm_range', 'left_forearm_mean',
                              'left_forearm_std', 'left_forearm_range', 'torso_tilt_mean', 'torso_tilt_std',
                              'torso_tilt_range', 'right_elbow_mean', 'right_elbow_std', 'right_elbow_range',
                              'left_elbow_mean', 'left_elbow_std', 'left_elbow_range', 'pose_symmetry_mean',
                              'pose_symmetry_std', 'pose_symmetry_min', 'hand_separation_mean', 'hand_separation_max',
                              'hand_separation_std',
                              # 'hand_height_correlation',
                              'time_in_high_zone', 'time_in_mid_zone', 'time_in_low_zone', 
                              # New person-relative features
                              'right_hand_mean_height', 'right_hand_max_height', 'right_hand_height_range',
                              'right_hand_mean_lateral_reach', 'right_hand_max_lateral_reach',
                              'right_hand_mean_elevation', 'right_hand_max_elevation', 'right_hand_overhead_time',
                              'left_hand_mean_height', 'left_hand_max_height', 'left_hand_height_range', 
                              'left_hand_mean_lateral_reach', 'left_hand_max_lateral_reach',
                              'left_hand_mean_elevation', 'left_hand_max_elevation', 'left_hand_overhead_time',
                              'combined_lateral_range', 'mean_lateral_activity',
                              'pose_stability', 'movement_complexity', 'bilateral_coordination']

    def generate_window(self, raw_data_file: str) -> List[Dict]:
        """
        Generate a window of pose data from a raw data file.

        Args:
            raw_data_file: Path to the raw pose data file

        Returns:
            List of frame dictionaries containing pose keypoints
        """
        # Load raw data file
        instance_data = pickle.load(open(raw_data_file, 'rb'))
        timestamps = np.array([xr[0] for xr in instance_data])

        pose_frames = []
        for instance_ts, pose_data in tqdm.tqdm(instance_data, desc="Loading pose data"):
            # Ensure pose_data is properly formatted
            if isinstance(pose_data, dict):
                pose_data = pose_data['keypoints']

            # Add timestamp to frame data
            frame_dict = {
                'timestamp': instance_ts,
                'keypoints': pose_data
            }
            pose_frames.append(pose_data)

        return pose_frames

    # def generate_timestamp_windows(self, raw_data_file: str):
    #     """
    #     Generate a window of pose data from a raw data file.

    #     Args:
    #         raw_data_file: Path to the raw pose data file

    #     Returns:
    #         List of frame dictionaries containing pose keypoints
    #     """
    #     # Load raw data file
    #     instance_data = pickle.load(open(raw_data_file, 'rb'))
    #     timestamps = np.array([xr[0] for xr in instance_data])

    #     pose_frames = []
    #     for instance_ts, pose_data in tqdm.tqdm(instance_data, desc="Loading pose data"):
    #         # Ensure pose_data is properly formatted
    #         if isinstance(pose_data, dict):
    #             pose_data = pose_data['keypoints']

    #         # Add timestamp to frame data
    #         frame_dict = {
    #             'timestamp': instance_ts,
    #             'keypoints': pose_data
    #         }
    #         pose_frames.append(pose_data)
    #     pose_features = self.extract_training_features(pose_frames,roll_size=1)
    #     return timestamps, pose_features

    def generate_clustering_window(self, raw_data_file: str) -> List[Dict]:
        """
        Generate a window specifically for clustering.
        Wrapper around generate_window for consistency with other implementations.

        Args:
            raw_data_file: Path to the raw pose data file

        Returns:
            List of frame dictionaries containing pose keypoints
        """
        return self.generate_window(raw_data_file)

    def generate_training_features(self, windows: List[List[np.ndarray]], window_ids: List[str]) -> Tuple[
        List[np.ndarray], List[str]]:
        """
        Generate features for multiple windows for training purposes.

        Args:
            windows: List of pose data windows, each containing list of frames
            window_ids: List of window identifiers

        Returns:
            Tuple containing:
                - List of feature arrays
                - List of corresponding window IDs
        """
        if len(windows) != len(window_ids):
            raise ValueError("Number of windows and window IDs must match")

        # Extract features for each window
        features_list = []
        valid_ids = []

        for window_idx, (window, window_id) in tqdm.tqdm(enumerate(zip(windows, window_ids)), total=len(windows), desc="Extracting pose features"):
            features = self.extract_training_features(window)
            features_list.append(features)
            valid_ids.append(window_id)

        return features_list, valid_ids

    def extract_training_features(self, window: List[Dict], roll_size: int = 20) -> np.ndarray:
        """
        Extract features from a single window sequence for training.

        Args:
            window: List of pose frame dictionaries
            roll_size: Size of rolling window for feature aggregation

        Returns:
            Array of features shaped (n_frames, n_features)
        """
        # First get raw pose vectors for each frame
        frame_features = []
        for frame in window:
            if frame.shape[0] == 0:
                # Empty frame, add zeros
                frame_features.append(np.zeros(36))
                continue
            frame = frame[0]
            # Flatten key joint coordinates with confidence
            pose_vec = []
            for joint_idx in range(17):  # All joints
                if frame[joint_idx][2] > PoseConstants.MIN_CONFIDENCE:
                    pose_vec.extend(frame[joint_idx][:2])  # x, y, confidence
                else:
                    pose_vec.extend([0, 0, 0][:2])  # Zero for low confidence joints
            frame_features.append(pose_vec)

        # Convert to array
        window_feature_vec = np.array(frame_features)

        # Apply rolling window aggregation if needed
        if window_feature_vec.shape[0] > roll_size:
            # Use sliding window and take maximum values
            training_features = sliding_window_view(window_feature_vec, roll_size, axis=0).max(axis=2)
        else:
            # For short sequences, just sum up the features
            training_features = window_feature_vec.sum(axis=0).reshape(1, -1)

        return training_features

    # def extract_training_features(self, window: List[np.ndarray], roll_size: int = 20) -> np.ndarray:
    #     """
    #     Extract features from a single window sequence for training.
    #
    #     Args:
    #         window: List of pose frames
    #         roll_size: Size of rolling window for feature aggregation
    #
    #     Returns:
    #         Array of features
    #     """
    #     # Get features using existing extraction method
    #     feature_array, _ = self.extract_features(window)
    #
    #     # Reshape for rolling window if needed
    #     if (len(feature_array) > roll_size) & (len(feature_array.shape) > 1):
    #         features = sliding_window_view(feature_array.reshape(1, -1), roll_size, axis=1).max(axis=2)
    #     else:
    #         features = feature_array.reshape(1, -1)
    #
    #     return features

    def extract_features(self, window:List[np.ndarray]) -> Dict[str, float]:
        """Extract features from a window of pose data"""
        # if len(window) < PoseConstants.MIN_FRAMES:
        #     raise ValueError(f"Window too short. Minimum {PoseConstants.MIN_FRAMES} frames required.")

        # Initialize features dictionary
        features = {}

        # Get temporal segments
        segments = self._segment_sequence(window)

        # Extract all feature types
        features.update(self._extract_temporal_features(segments))
        features.update(self._extract_spatial_features(segments))
        features.update(self._extract_zone_features(segments))
        features.update(self._extract_complexity_features(segments))

        return features

    def _segment_sequence(self, window: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Split window into temporal segments (YOLO/COCO: 17 keypoints)"""
        single_person_data = [(frame[0] if (frame.shape[0] > 0) else np.zeros((17, 3))) for frame in window]
        n_segments = PoseConstants.TIME_SEGMENTS
        segment_size = (len(single_person_data) // n_segments)
        if segment_size == 0:
            return [single_person_data]
        return [single_person_data[i:i + segment_size] for i in range(0, len(single_person_data), segment_size)]

    def _extract_temporal_features(self, segments: List[List[np.ndarray]]) -> Dict[str, float]:
        """Extract comprehensive temporal motion features including full body motion analysis"""
        features = {}

        # Track velocities for key joints
        joint_velocities = {
            'hands': {'right': [], 'left': []},
            'elbows': {'right': [], 'left': []},
            'head': [],  # Using nose as reference
            'shoulders': {'right': [], 'left': []}
        }

        # Track accelerations
        joint_accelerations = {
            'hands': {'right': [], 'left': []},
            'elbows': {'right': [], 'left': []},
            'head': []
        }

        # Track body orientation
        torso_orientations = []

        # Track static vs dynamic states
        motion_states = []  # 1 for dynamic, 0 for static

        for segment in segments:
            segment_length = len(segment)

            for i in range(segment_length - 1):
                curr_frame = segment[i]
                next_frame = segment[i + 1]

                # 1. Compute velocities for all key joints
                self._compute_joint_velocity(curr_frame, next_frame, PoseConstants.R_WRIST,
                                             joint_velocities['hands']['right'])
                self._compute_joint_velocity(curr_frame, next_frame, PoseConstants.L_WRIST,
                                             joint_velocities['hands']['left'])
                self._compute_joint_velocity(curr_frame, next_frame, PoseConstants.R_ELBOW,
                                             joint_velocities['elbows']['right'])
                self._compute_joint_velocity(curr_frame, next_frame, PoseConstants.L_ELBOW,
                                             joint_velocities['elbows']['left'])
                self._compute_joint_velocity(curr_frame, next_frame, PoseConstants.NOSE,
                                             joint_velocities['head'])
                self._compute_joint_velocity(curr_frame, next_frame, PoseConstants.R_SHOULDER,
                                             joint_velocities['shoulders']['right'])
                self._compute_joint_velocity(curr_frame, next_frame, PoseConstants.L_SHOULDER,
                                             joint_velocities['shoulders']['left'])

                # 2. Compute body orientation
                if i < segment_length - 2:  # Need three frames for acceleration
                    next_next_frame = segment[i + 2]

                    # Compute accelerations for key joints
                    for joint_type in ['hands', 'elbows', 'head']:
                        if joint_type == 'head':
                            self._compute_joint_acceleration(curr_frame, next_frame, next_next_frame,
                                                             PoseConstants.NOSE, joint_accelerations[joint_type])
                        else:
                            self._compute_joint_acceleration(curr_frame, next_frame, next_next_frame,
                                                             PoseConstants.R_WRIST if joint_type == 'hands' else PoseConstants.R_ELBOW,
                                                             joint_accelerations[joint_type]['right'])
                            self._compute_joint_acceleration(curr_frame, next_frame, next_next_frame,
                                                             PoseConstants.L_WRIST if joint_type == 'hands' else PoseConstants.L_ELBOW,
                                                             joint_accelerations[joint_type]['left'])

                # 3. Compute torso orientation
                orientation = self._compute_torso_orientation(curr_frame)
                if orientation is not None:
                    torso_orientations.append(orientation)

                # 4. Detect static vs dynamic poses
                total_motion = 0
                confidence_count = 0
                for joint_idx in range(17):  # Check all joints
                    if (curr_frame[joint_idx][2] > PoseConstants.MIN_CONFIDENCE and
                            next_frame[joint_idx][2] > PoseConstants.MIN_CONFIDENCE):
                        total_motion += np.linalg.norm(next_frame[joint_idx][:2] - curr_frame[joint_idx][:2])
                        confidence_count += 1

                if confidence_count > 0:
                    avg_motion = total_motion / confidence_count
                    motion_states.append(1 if avg_motion > 2.0 else 0)  # Threshold for static vs dynamic

        # Compute final temporal features
        features.update(self._compute_velocity_features(joint_velocities))
        features.update(self._compute_acceleration_features(joint_accelerations))
        features.update(self._compute_orientation_features(torso_orientations))
        features.update(self._compute_motion_state_features(motion_states))

        return features

    def _compute_joint_velocity(self, curr_frame: Dict, next_frame: Dict,
                                joint_idx: int, velocity_list: List[float]) -> None:
        """Compute velocity for a specific joint if confidence threshold is met (YOLO/COCO)"""
        if (curr_frame[joint_idx][2] > PoseConstants.MIN_CONFIDENCE and
                next_frame[joint_idx][2] > PoseConstants.MIN_CONFIDENCE):
            velocity = np.linalg.norm(next_frame[joint_idx][:2] - curr_frame[joint_idx][:2])
            velocity_list.append(velocity)

    def _compute_joint_acceleration(self, curr_frame: Dict, next_frame: Dict,
                                    next_next_frame: Dict, joint_idx: int,
                                    acceleration_list: List[float]) -> None:
        """Compute acceleration for a specific joint if confidence threshold is met (YOLO/COCO)"""
        if (curr_frame[joint_idx][2] > PoseConstants.MIN_CONFIDENCE and
                next_frame[joint_idx][2] > PoseConstants.MIN_CONFIDENCE and
                next_next_frame[joint_idx][2] > PoseConstants.MIN_CONFIDENCE):
            vel1 = next_frame[joint_idx][:2] - curr_frame[joint_idx][:2]
            vel2 = next_next_frame[joint_idx][:2] - next_frame[joint_idx][:2]
            acceleration = np.linalg.norm(vel2 - vel1)
            acceleration_list.append(acceleration)

    def _compute_torso_orientation(self, frame: Dict) -> Optional[float]:
        """Compute torso orientation relative to vertical axis (YOLO/COCO: no NECK, use shoulders/hips)"""
        if (frame[PoseConstants.L_SHOULDER][2] > PoseConstants.MIN_CONFIDENCE and
                frame[PoseConstants.R_SHOULDER][2] > PoseConstants.MIN_CONFIDENCE and
                frame[PoseConstants.L_HIP][2] > PoseConstants.MIN_CONFIDENCE and
                frame[PoseConstants.R_HIP][2] > PoseConstants.MIN_CONFIDENCE):
            shoulder_center = (frame[PoseConstants.L_SHOULDER][:2] + frame[PoseConstants.R_SHOULDER][:2]) / 2
            hip_center = (frame[PoseConstants.L_HIP][:2] + frame[PoseConstants.R_HIP][:2]) / 2
            orientation = np.arctan2(shoulder_center[0] - hip_center[0], shoulder_center[1] - hip_center[1])
            return np.degrees(orientation)
        return None

    def _compute_velocity_features(self, joint_velocities: Dict) -> Dict[str, float]:
        """Compute features from joint velocities"""
        features = {}

        # Compute statistics for each joint
        for joint_type, data in joint_velocities.items():
            if isinstance(data, dict):  # Bilateral joints
                all_vels = data['right'] + data['left']
                prefix = f"{joint_type}_"
            else:  # Single joints (head)
                all_vels = data
                prefix = f"{joint_type}_"

            if all_vels:
                features[f"{prefix}mean_velocity"] = np.mean(all_vels)
                features[f"{prefix}max_velocity"] = np.max(all_vels)
                features[f"{prefix}velocity_std"] = np.std(all_vels)
                features[f"{prefix}direction_changes"] = len(self._count_direction_changes(all_vels))

        return features

    def _compute_acceleration_features(self, joint_accelerations: Dict) -> Dict[str, float]:
        """Compute features from joint accelerations"""
        features = {}

        for joint_type, data in joint_accelerations.items():
            if isinstance(data, dict):  # Bilateral joints
                all_accs = data['right'] + data['left']
                prefix = f"{joint_type}_"
            else:  # Single joints (head)
                all_accs = data
                prefix = f"{joint_type}_"

            if all_accs:
                features[f"{prefix}mean_acceleration"] = np.mean(all_accs)
                features[f"{prefix}max_acceleration"] = np.max(all_accs)
                features[f"{prefix}acceleration_std"] = np.std(all_accs)

        return features

    def _compute_orientation_features(self, orientations: List[float]) -> Dict[str, float]:
        """Compute features from torso orientations"""
        features = {}

        if orientations:
            # Basic statistics
            features['mean_orientation'] = np.mean(orientations)
            features['orientation_range'] = np.ptp(orientations)

            # Orientation changes
            orientation_changes = np.abs(np.diff(orientations))
            features['total_rotation'] = np.sum(orientation_changes)
            features['max_rotation'] = np.max(orientation_changes) if len(orientation_changes) > 0 else 0
        else:
            features['mean_orientation'] = 0
            features['orientation_range'] = 0
            features['total_rotation'] = 0
            features['max_rotation'] = 0

        return features

    def _compute_motion_state_features(self, motion_states: List[int]) -> Dict[str, float]:
        """Compute features from static vs dynamic states"""
        features = {}

        if motion_states:
            # Compute proportion of time in dynamic state
            features['dynamic_ratio'] = np.mean(motion_states)

            # Find longest static and dynamic sequences
            static_sequences = self._find_longest_sequence(motion_states, 0)
            dynamic_sequences = self._find_longest_sequence(motion_states, 1)

            features['longest_static_duration'] = static_sequences
            features['longest_dynamic_duration'] = dynamic_sequences

            # Compute number of transitions between states
            features['state_transitions'] = np.sum(np.abs(np.diff(motion_states)))
        else:
            features['dynamic_ratio'] = 0
            features['longest_static_duration'] = 0
            features['longest_dynamic_duration'] = 0
            features['state_transitions'] = 0

        return features

    def _find_longest_sequence(self, states: List[int], target_state: int) -> int:
        """Find the longest continuous sequence of a target state"""
        if not states:
            return 0

        current_seq = 0
        longest_seq = 0

        for state in states:
            if state == target_state:
                current_seq += 1
                longest_seq = max(longest_seq, current_seq)
            else:
                current_seq = 0

        return longest_seq

        return features

    def _extract_spatial_features(self, segments: List[List[Dict]]) -> Dict[str, float]:
        """Extract comprehensive spatial configuration features"""
        features = {}

        # Collectors for sequence-level features
        com_trajectory = []
        working_zone_stats = {
            'right_hand': {'x': [], 'y': [], 'reach': []},
            'left_hand': {'x': [], 'y': [], 'reach': []},
        }
        pose_stats = {
            'limb_lengths': defaultdict(list),
            'pose_angles': defaultdict(list),
            'symmetry': [],
            'torso_tilt': []
        }

        # Process each frame in segments
        for segment in segments:
            for frame in segment:
                # 1. Center of Mass
                com = self._compute_body_com(frame)
                if com is not None:
                    com_trajectory.append(com)

                # 2. Working Zone
                zone_metrics = self._compute_working_zone_metrics(frame)
                for hand in ['right', 'left']:
                    if f'{hand}_hand_horizontal' in zone_metrics:
                        working_zone_stats[f'{hand}_hand']['x'].append(zone_metrics[f'{hand}_hand_horizontal'])
                        working_zone_stats[f'{hand}_hand']['y'].append(zone_metrics[f'{hand}_hand_vertical'])
                        working_zone_stats[f'{hand}_hand']['reach'].append(zone_metrics[f'{hand}_hand_distance'])

                # 3. Limb Configuration
                limb_lengths = self._compute_relative_limb_lengths(frame)
                for limb, length in limb_lengths.items():
                    pose_stats['limb_lengths'][limb].append(length)

                # 4. Pose Angles
                angles = self._compute_pose_angles(frame)
                for angle_name, angle_val in angles.items():
                    pose_stats['pose_angles'][angle_name].append(angle_val)
                if 'torso_tilt' in angles:
                    pose_stats['torso_tilt'].append(angles['torso_tilt'])

                # 5. Symmetry
                symmetry = self._compute_pose_symmetry(frame)
                pose_stats['symmetry'].append(symmetry)

        # Compute final features

        # 1. Center of Mass Features
        if com_trajectory:
            com_trajectory = np.array(com_trajectory)
            features.update({
                'com_vertical_range': np.ptp(com_trajectory[:, 1]),
                'com_horizontal_range': np.ptp(com_trajectory[:, 0]),
                'com_path_length': np.sum(np.linalg.norm(np.diff(com_trajectory, axis=0), axis=1)),
                'com_mean_height': np.mean(com_trajectory[:, 1]),
                'com_std_horizontal': np.std(com_trajectory[:, 0])
            })

        # 2. Working Zone Features
        for hand in ['right', 'left']:
            hand_stats = working_zone_stats[f'{hand}_hand']
            if hand_stats['reach']:
                prefix = f'{hand}_hand'
                features.update({
                    f'{prefix}_max_reach': np.max(hand_stats['reach']),
                    f'{prefix}_mean_reach': np.mean(hand_stats['reach']),
                    f'{prefix}_vertical_range': np.ptp(hand_stats['y']),
                    f'{prefix}_horizontal_range': np.ptp(hand_stats['x']),
                    f'{prefix}_mean_height': np.mean(hand_stats['y']),
                    f'{prefix}_workspace_area': np.ptp(hand_stats['x']) * np.ptp(hand_stats['y'])
                })

        # 3. Limb Length Features
        for limb, lengths in pose_stats['limb_lengths'].items():
            if lengths:
                features.update({
                    f'{limb}_mean': np.mean(lengths),
                    f'{limb}_std': np.std(lengths),
                    f'{limb}_range': np.ptp(lengths)
                })

        # 4. Pose Angle Features
        for angle_name, angles in pose_stats['pose_angles'].items():
            if angles:
                features.update({
                    f'{angle_name}_mean': np.mean(angles),
                    f'{angle_name}_std': np.std(angles),
                    f'{angle_name}_range': np.ptp(angles)
                })

        # 5. Posture Features
        if pose_stats['torso_tilt']:
            features.update({
                'torso_tilt_mean': np.mean(pose_stats['torso_tilt']),
                'torso_tilt_std': np.std(pose_stats['torso_tilt']),
                'torso_tilt_range': np.ptp(pose_stats['torso_tilt'])
            })

        if pose_stats['symmetry']:
            features.update({
                'pose_symmetry_mean': np.mean(pose_stats['symmetry']),
                'pose_symmetry_std': np.std(pose_stats['symmetry']),
                'pose_symmetry_min': np.min(pose_stats['symmetry'])
            })

        # 6. Bimanual Coordination
        if (working_zone_stats['right_hand']['x'] and
                working_zone_stats['left_hand']['x']):
            # Compute hand separation statistics
            hand_separations = np.array([
                np.sqrt(
                    (rx - lx) ** 2 + (ry - ly) ** 2
                ) for rx, lx, ry, ly in zip(
                    working_zone_stats['right_hand']['x'],
                    working_zone_stats['left_hand']['x'],
                    working_zone_stats['right_hand']['y'],
                    working_zone_stats['left_hand']['y']
                )
            ])
            features.update({
                'hand_separation_mean': np.mean(hand_separations),
                'hand_separation_max': np.max(hand_separations),
                'hand_separation_std': np.std(hand_separations)
            })

            # Compute hand height correlation
            # height_corr = np.corrcoef(
            #     working_zone_stats['right_hand']['y'],
            #     working_zone_stats['left_hand']['y']
            # )[0, 1]
            # features['hand_height_correlation'] = height_corr

        return features

    def _extract_zone_features(self, segments: List[List[Dict]]) -> Dict[str, float]:
        """Extract working zone features using person-relative positioning"""
        features = {}
        
        # Collect relative positioning data for both hands
        hand_metrics = {
            'right': {'height_relative': [], 'lateral_reach': [], 'shoulder_elevation': []},
            'left': {'height_relative': [], 'lateral_reach': [], 'shoulder_elevation': []}
        }
        
        zone_counts = {
            'high': 0,    # Above shoulder level (overhead reach)
            'mid': 0,     # Shoulder to mid-torso level (normal work) 
            'low': 0      # Mid-torso to below hip (below counter)
        }
        
        total_hand_observations = 0
        
        for segment in segments:
            for frame in segment:
                # Analyze right wrist
                right_metrics = self._compute_person_relative_position(frame, PoseConstants.R_WRIST)
                if right_metrics:
                    hand_metrics['right']['height_relative'].append(right_metrics['height_relative'])
                    hand_metrics['right']['lateral_reach'].append(right_metrics['lateral_reach'])
                    hand_metrics['right']['shoulder_elevation'].append(right_metrics['shoulder_elevation'])
                    
                    # Classify into zones based on height_relative
                    height_rel = right_metrics['height_relative']
                    if PoseConstants.ZONE_THRESHOLDS['high'][0] <= height_rel < PoseConstants.ZONE_THRESHOLDS['high'][1]:
                        zone_counts['high'] += 1
                    elif PoseConstants.ZONE_THRESHOLDS['mid'][0] <= height_rel < PoseConstants.ZONE_THRESHOLDS['mid'][1]:
                        zone_counts['mid'] += 1
                    elif PoseConstants.ZONE_THRESHOLDS['low'][0] <= height_rel < PoseConstants.ZONE_THRESHOLDS['low'][1]:
                        zone_counts['low'] += 1
                    total_hand_observations += 1
                
                # Analyze left wrist  
                left_metrics = self._compute_person_relative_position(frame, PoseConstants.L_WRIST)
                if left_metrics:
                    hand_metrics['left']['height_relative'].append(left_metrics['height_relative'])
                    hand_metrics['left']['lateral_reach'].append(left_metrics['lateral_reach'])
                    hand_metrics['left']['shoulder_elevation'].append(left_metrics['shoulder_elevation'])
                    
                    # Classify into zones based on height_relative
                    height_rel = left_metrics['height_relative']
                    if PoseConstants.ZONE_THRESHOLDS['high'][0] <= height_rel < PoseConstants.ZONE_THRESHOLDS['high'][1]:
                        zone_counts['high'] += 1
                    elif PoseConstants.ZONE_THRESHOLDS['mid'][0] <= height_rel < PoseConstants.ZONE_THRESHOLDS['mid'][1]:
                        zone_counts['mid'] += 1
                    elif PoseConstants.ZONE_THRESHOLDS['low'][0] <= height_rel < PoseConstants.ZONE_THRESHOLDS['low'][1]:
                        zone_counts['low'] += 1
                    total_hand_observations += 1

        # Compute zone time features (normalized by total hand observations)
        if total_hand_observations > 0:
            features['time_in_high_zone'] = zone_counts['high'] / total_hand_observations
            features['time_in_mid_zone'] = zone_counts['mid'] / total_hand_observations  
            features['time_in_low_zone'] = zone_counts['low'] / total_hand_observations
        else:
            features['time_in_high_zone'] = 0
            features['time_in_mid_zone'] = 0
            features['time_in_low_zone'] = 0

        # Compute aggregate hand positioning features
        for hand in ['right', 'left']:
            prefix = f'{hand}_hand'
            
            if hand_metrics[hand]['height_relative']:
                # Height statistics
                features[f'{prefix}_mean_height'] = np.mean(hand_metrics[hand]['height_relative'])
                features[f'{prefix}_max_height'] = np.max(hand_metrics[hand]['height_relative'])
                features[f'{prefix}_height_range'] = np.ptp(hand_metrics[hand]['height_relative'])
                
                # Lateral reach statistics
                features[f'{prefix}_mean_lateral_reach'] = np.mean(hand_metrics[hand]['lateral_reach'])
                features[f'{prefix}_max_lateral_reach'] = np.max(hand_metrics[hand]['lateral_reach'])
                
                # Shoulder elevation statistics (positive = above shoulder)
                elevations = hand_metrics[hand]['shoulder_elevation']
                features[f'{prefix}_mean_elevation'] = np.mean(elevations)
                features[f'{prefix}_max_elevation'] = np.max(elevations)
                features[f'{prefix}_overhead_time'] = np.mean([1 if e > 0.1 else 0 for e in elevations])  # Time above shoulder
                
        # Combined hand features
        all_lateral_reaches = (hand_metrics['right']['lateral_reach'] + 
                              hand_metrics['left']['lateral_reach'])
        if all_lateral_reaches:
            features['combined_lateral_range'] = np.ptp(all_lateral_reaches)
            features['mean_lateral_activity'] = np.mean(all_lateral_reaches)
        else:
            features['combined_lateral_range'] = 0
            features['mean_lateral_activity'] = 0

        return features

    def _extract_complexity_features(self, segments: List[List[Dict]]) -> Dict[str, float]:
        """Extract pose complexity features"""
        features = {}

        # Pose stability (consistency of joint positions)
        joint_variations = []
        for joint_idx in range(17):  # All joints
            positions = []
            for segment in segments:
                for frame in segment:
                    if frame[joint_idx][2] > PoseConstants.MIN_CONFIDENCE:
                        positions.append(frame[joint_idx][:2])
            if positions:
                joint_variations.append(np.std(positions, axis=0).mean())

        features['pose_stability'] = 1.0 / (1.0 + np.mean(joint_variations)) if joint_variations else 0

        # Movement complexity (variation in velocities)
        velocity_entropies = []
        for segment in segments:
            velocities = self._compute_joint_velocities(segment)
            if velocities:
                velocity_entropies.append(self._compute_entropy(velocities))

        features['movement_complexity'] = np.mean(velocity_entropies) if velocity_entropies else 0

        # Bilateral coordination (correlation between left and right side movements)
        features['bilateral_coordination'] = self._compute_bilateral_correlation(segments)

        return features

    def _compute_arm_extension(self, frame: Dict, side: str) -> float:
        """Compute arm extension ratio"""
        if side == 'right':
            shoulder, elbow, wrist = PoseConstants.R_SHOULDER, PoseConstants.R_ELBOW, PoseConstants.R_WRIST
        else:
            shoulder, elbow, wrist = PoseConstants.L_SHOULDER, PoseConstants.L_ELBOW, PoseConstants.L_WRIST

        if all(frame[i][2] > PoseConstants.MIN_CONFIDENCE for i in [shoulder, elbow, wrist]):
            # Actual distance from shoulder to wrist
            actual_dist = np.linalg.norm(frame[wrist][:2] - frame[shoulder][:2])
            # Maximum possible distance (sum of upper arm and forearm lengths)
            upper_arm = np.linalg.norm(frame[elbow][:2] - frame[shoulder][:2])
            forearm = np.linalg.norm(frame[wrist][:2] - frame[elbow][:2])
            max_dist = upper_arm + forearm
            return actual_dist / max_dist if max_dist > 0 else 0
        return 0

    def _compute_torso_angle(self, frame: Dict) -> float:
        """Compute torso angle from vertical"""
        neck = frame[PoseConstants.NOSE][:2]
        hip_center = (frame[PoseConstants.R_HIP][:2] + frame[PoseConstants.L_HIP][:2]) / 2

        # Angle from vertical (y-axis)
        vector = neck - hip_center
        angle = np.arctan2(vector[0], vector[1])  # from vertical
        return np.degrees(angle)

    def _compute_joint_velocities(self, segment: List[Dict]) -> List[float]:
        """Compute joint velocities for a segment"""
        velocities = []
        for i in range(len(segment) - 1):
            for joint_idx in range(17):
                if (segment[i][joint_idx][2] > PoseConstants.MIN_CONFIDENCE and
                        segment[i + 1][joint_idx][2] > PoseConstants.MIN_CONFIDENCE):
                    vel = np.linalg.norm(
                        segment[i + 1][joint_idx][:2] - segment[i][joint_idx][:2]
                    )
                    velocities.append(vel)
        return velocities

    def _compute_entropy(self, values: List[float], bins: int = 10) -> float:
        """Compute entropy of a distribution"""
        if not values:
            return 0
        hist, _ = np.histogram(values, bins=bins, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0

    def _compute_bilateral_correlation(self, segments: List[List[Dict]]) -> float:
        """Compute correlation between left and right side movements"""
        right_traj = []
        left_traj = []

        for segment in segments:
            for frame in segment:
                if (frame[PoseConstants.R_WRIST][2] > PoseConstants.MIN_CONFIDENCE and
                        frame[PoseConstants.L_WRIST][2] > PoseConstants.MIN_CONFIDENCE):
                    right_traj.append(frame[PoseConstants.R_WRIST][:2])
                    left_traj.append(frame[PoseConstants.L_WRIST][:2])

        if len(right_traj) > 1:
            right_vel = np.diff(right_traj, axis=0)
            left_vel = np.diff(left_traj, axis=0)

            if len(right_vel) > 0:
                correlation = np.corrcoef(
                    np.linalg.norm(right_vel, axis=1),
                    np.linalg.norm(left_vel, axis=1)
                )[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0

        return 0

    def _count_direction_changes(self, velocities: List[float]) -> List[int]:
        """Count number of velocity direction changes"""
        if len(velocities) < 2:
            return []

        # Smooth velocities to reduce noise
        if len(velocities) > 5:
            velocities = savgol_filter(velocities, 5, 2)

        # Find zero crossings in acceleration
        acceleration = np.diff(velocities)
        direction_changes = np.where(np.diff(np.signbit(acceleration)))[0]
        return direction_changes

    # Additional helper methods for spatial features - add to PoseFeatures class

    def _compute_relative_limb_lengths(self, frame: Dict) -> Dict[str, float]:
        """Compute relative lengths of limbs normalized by shoulder width (YOLO/COCO)"""
        lengths = {}
        # Get shoulder width for normalization
        if (frame[PoseConstants.L_SHOULDER][2] > PoseConstants.MIN_CONFIDENCE and
                frame[PoseConstants.R_SHOULDER][2] > PoseConstants.MIN_CONFIDENCE):
            shoulder_width = np.linalg.norm(
                frame[PoseConstants.L_SHOULDER][:2] - frame[PoseConstants.R_SHOULDER][:2]
            )
            # Left arm segments
            if all(frame[j][2] > PoseConstants.MIN_CONFIDENCE for j in
                   [PoseConstants.L_SHOULDER, PoseConstants.L_ELBOW, PoseConstants.L_WRIST]):
                upper_arm = np.linalg.norm(
                    frame[PoseConstants.L_ELBOW][:2] - frame[PoseConstants.L_SHOULDER][:2]
                )
                forearm = np.linalg.norm(
                    frame[PoseConstants.L_WRIST][:2] - frame[PoseConstants.L_ELBOW][:2]
                )
                lengths['left_upper_arm'] = upper_arm / shoulder_width
                lengths['left_forearm'] = forearm / shoulder_width
            # Right arm segments
            if all(frame[j][2] > PoseConstants.MIN_CONFIDENCE for j in
                   [PoseConstants.R_SHOULDER, PoseConstants.R_ELBOW, PoseConstants.R_WRIST]):
                upper_arm = np.linalg.norm(
                    frame[PoseConstants.R_ELBOW][:2] - frame[PoseConstants.R_SHOULDER][:2]
                )
                forearm = np.linalg.norm(
                    frame[PoseConstants.R_WRIST][:2] - frame[PoseConstants.R_ELBOW][:2]
                )
                lengths['right_upper_arm'] = upper_arm / shoulder_width
                lengths['right_forearm'] = forearm / shoulder_width
        return lengths

    def _compute_body_com(self, frame: Dict) -> Optional[np.ndarray]:
        """
        Compute approximate body center of mass using weighted joint positions (YOLO/COCO)
        Returns None if insufficient joints are detected
        """
        # Joint weights based on approximate segment mass proportions
        joint_weights = {
            PoseConstants.L_SHOULDER: 0.1,  # Left upper arm
            PoseConstants.R_SHOULDER: 0.1,  # Right upper arm
            PoseConstants.L_ELBOW: 0.05,    # Left forearm
            PoseConstants.R_ELBOW: 0.05,    # Right forearm
            PoseConstants.L_HIP: 0.25,      # Left trunk half
            PoseConstants.R_HIP: 0.25,      # Right trunk half
            PoseConstants.NOSE: 0.2         # Head (no NECK in YOLO)
        }
        weighted_positions = []
        total_weight = 0
        for joint, weight in joint_weights.items():
            if frame[joint][2] > PoseConstants.MIN_CONFIDENCE:
                weighted_positions.append(frame[joint][:2] * weight)
                total_weight += weight
        if total_weight > 0.6:  # At least 60% of body mass accounted for
            return np.sum(weighted_positions, axis=0) / total_weight
        return None

    def _compute_pose_angles(self, frame: Dict) -> Dict[str, float]:
        """Compute key pose angles (YOLO/COCO)"""
        angles = {}
        # Torso angle from vertical
        if all(frame[j][2] > PoseConstants.MIN_CONFIDENCE for j in
               [PoseConstants.L_SHOULDER, PoseConstants.R_SHOULDER, PoseConstants.L_HIP, PoseConstants.R_HIP]):
            shoulder_center = (frame[PoseConstants.L_SHOULDER][:2] + frame[PoseConstants.R_SHOULDER][:2]) / 2
            hip_center = (frame[PoseConstants.L_HIP][:2] + frame[PoseConstants.R_HIP][:2]) / 2
            torso_vector = shoulder_center - hip_center
            angles['torso_tilt'] = np.degrees(np.arctan2(torso_vector[0], torso_vector[1]))
        # Elbow angles
        if all(frame[j][2] > PoseConstants.MIN_CONFIDENCE for j in
               [PoseConstants.R_SHOULDER, PoseConstants.R_ELBOW, PoseConstants.R_WRIST]):
            v1 = frame[PoseConstants.R_ELBOW][:2] - frame[PoseConstants.R_SHOULDER][:2]
            v2 = frame[PoseConstants.R_WRIST][:2] - frame[PoseConstants.R_ELBOW][:2]
            angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
            angles['right_elbow'] = angle
        if all(frame[j][2] > PoseConstants.MIN_CONFIDENCE for j in
               [PoseConstants.L_SHOULDER, PoseConstants.L_ELBOW, PoseConstants.L_WRIST]):
            v1 = frame[PoseConstants.L_ELBOW][:2] - frame[PoseConstants.L_SHOULDER][:2]
            v2 = frame[PoseConstants.L_WRIST][:2] - frame[PoseConstants.L_ELBOW][:2]
            angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
            angles['left_elbow'] = angle
        return angles

    def _compute_working_zone_metrics(self, frame: Dict) -> Dict[str, float]:
        """Compute metrics related to working zone/area (YOLO/COCO)"""
        metrics = {}
        # Get hip center and shoulder width for normalization
        if all(frame[j][2] > PoseConstants.MIN_CONFIDENCE for j in
               [PoseConstants.L_HIP, PoseConstants.R_HIP,
                PoseConstants.L_SHOULDER, PoseConstants.R_SHOULDER]):
            hip_center = (frame[PoseConstants.L_HIP][:2] + frame[PoseConstants.R_HIP][:2]) / 2
            shoulder_width = np.linalg.norm(
                frame[PoseConstants.L_SHOULDER][:2] - frame[PoseConstants.R_SHOULDER][:2]
            )
            # Compute hand positions relative to hip center
            for side, wrist in [(PoseConstants.R_WRIST, 'right'),
                                (PoseConstants.L_WRIST, 'left')]:
                if frame[side][2] > PoseConstants.MIN_CONFIDENCE:
                    relative_pos = frame[side][:2] - hip_center
                    # Normalize by shoulder width
                    relative_pos = relative_pos / shoulder_width
                    metrics[f'{wrist}_hand_horizontal'] = relative_pos[0]
                    metrics[f'{wrist}_hand_vertical'] = relative_pos[1]
                    metrics[f'{wrist}_hand_distance'] = np.linalg.norm(relative_pos)
        return metrics

    def _compute_pose_symmetry(self, frame: Dict) -> float:
        """Compute symmetry between left and right body sides (YOLO/COCO)"""
        joint_pairs = [
            (PoseConstants.L_SHOULDER, PoseConstants.R_SHOULDER),
            (PoseConstants.L_ELBOW, PoseConstants.R_ELBOW),
            (PoseConstants.L_WRIST, PoseConstants.R_WRIST),
            (PoseConstants.L_HIP, PoseConstants.R_HIP)
        ]
        symmetry_scores = []
        for left, right in joint_pairs:
            if (frame[left][2] > PoseConstants.MIN_CONFIDENCE and
                    frame[right][2] > PoseConstants.MIN_CONFIDENCE):
                # Compare y-coordinates for height symmetry
                height_diff = abs(frame[left][1] - frame[right][1])
                x_diff = abs(abs(frame[left][0]) - abs(frame[right][0]))  # Compare distance from center
                # Convert differences to similarity scores (1 = perfectly symmetric)
                height_sym = 1 / (1 + height_diff / 100)  # Scale factor of 100 for pixels
                x_sym = 1 / (1 + x_diff / 100)
                symmetry_scores.append((height_sym + x_sym) / 2)
        return np.mean(symmetry_scores) if symmetry_scores else 0.0

    

    def calculate_frame_timestamps(self, start_time: datetime, end_time: datetime, num_frames: int) -> List[datetime]:
        """Calculate individual frame timestamps."""
        if num_frames <= 1:
            return [start_time]
        
        total_duration = (end_time - start_time).total_seconds()
        frame_interval = total_duration / (num_frames - 1)  # -1 because we include both start and end
        
        timestamps = []
        for i in range(num_frames):
            frame_time = start_time + timedelta(seconds=i * frame_interval)
            timestamps.append(frame_time)
        
        return timestamps

    def parse_filename_timestamps(self, filename: str):
        """Parse start and end timestamps from filename (same as VideoThermalFeaturizer)."""
        import os
        import re
        from datetime import datetime
        basename = os.path.basename(filename)
        pattern = r'.*-(\d{8}_\d{6})_(\d{8}_\d{6})\.mp4$'
        match = re.search(pattern, basename)
        if not match:
            raise ValueError(f"Filename {basename} doesn't match expected pattern: <prefix>_<start_time>_<end_time>.mp4")
        start_time_str, end_time_str = match.groups()
        start_time = datetime.strptime(start_time_str, '%Y%m%d_%H%M%S')
        end_time = datetime.strptime(end_time_str, '%Y%m%d_%H%M%S')
        return start_time, end_time

    def _compute_person_relative_position(self, frame: np.ndarray, joint_idx: int) -> Optional[Dict[str, float]]:
        """
        Compute joint position relative to person's body landmarks.
        Returns normalized positions relative to person height and torso center.
        
        Args:
            frame: Single pose frame (17, 3)
            joint_idx: Index of joint to analyze
            
        Returns:
            Dict with relative positioning metrics or None if insufficient landmarks
        """
        if frame.shape[0] == 0:
            return None
            
        pose = frame[0] if len(frame.shape) == 3 else frame
        
        # Check if target joint is confident
        if pose[joint_idx][2] <= PoseConstants.MIN_CONFIDENCE:
            return None
            
        # Get key reference points with confidence checks
        head_joints = [PoseConstants.NOSE]  # Could add eyes/ears if needed
        shoulder_joints = [PoseConstants.L_SHOULDER, PoseConstants.R_SHOULDER]
        hip_joints = [PoseConstants.L_HIP, PoseConstants.R_HIP]
        
        # Find best head reference (nose preferred)
        head_pos = None
        for joint in head_joints:
            if pose[joint][2] > PoseConstants.MIN_CONFIDENCE:
                head_pos = pose[joint][:2]
                break
                
        # Compute shoulder center
        shoulder_pos = None
        valid_shoulders = [pose[j][:2] for j in shoulder_joints 
                          if pose[j][2] > PoseConstants.MIN_CONFIDENCE]
        if len(valid_shoulders) >= 1:
            shoulder_pos = np.mean(valid_shoulders, axis=0)
            
        # Compute hip center  
        hip_pos = None
        valid_hips = [pose[j][:2] for j in hip_joints 
                     if pose[j][2] > PoseConstants.MIN_CONFIDENCE]
        if len(valid_hips) >= 1:
            hip_pos = np.mean(valid_hips, axis=0)
            
        # Need at least head/shoulder and hip for height reference
        if (head_pos is None and shoulder_pos is None) or hip_pos is None:
            return None
            
        # Use head if available, otherwise shoulder as top reference
        top_ref = head_pos if head_pos is not None else shoulder_pos
        
        # Compute person height (top reference to hip)
        person_height = abs(hip_pos[1] - top_ref[1])
        if person_height < 10:  # Avoid division by very small numbers
            return None
            
        # Get target joint position
        joint_pos = pose[joint_idx][:2]
        
        # Compute relative metrics
        metrics = {}
        
        # 1. Vertical position relative to person height
        # 0.0 = top reference level, 1.0 = hip level, >1.0 = below hip
        vertical_relative = (joint_pos[1] - top_ref[1]) / person_height
        metrics['height_relative'] = max(0.0, vertical_relative)  # Clamp to positive
        
        # 2. Horizontal distance from torso center
        torso_center_x = (shoulder_pos[0] + hip_pos[0]) / 2 if shoulder_pos is not None else hip_pos[0]
        horizontal_offset = abs(joint_pos[0] - torso_center_x)
        # Normalize by shoulder width if available
        if shoulder_pos is not None and len(valid_shoulders) == 2:
            shoulder_width = abs(valid_shoulders[1][0] - valid_shoulders[0][0])
            if shoulder_width > 5:
                metrics['lateral_reach'] = horizontal_offset / shoulder_width
            else:
                metrics['lateral_reach'] = 0.0
        else:
            # Fallback: normalize by person height
            metrics['lateral_reach'] = horizontal_offset / person_height
            
        # 3. Distance from shoulder (for reach analysis)
        if shoulder_pos is not None:
            shoulder_distance = np.linalg.norm(joint_pos - shoulder_pos)
            metrics['shoulder_distance'] = shoulder_distance / person_height
        else:
            metrics['shoulder_distance'] = 0.0
            
        # 4. Elevation relative to shoulder (positive = above shoulder)
        if shoulder_pos is not None:
            elevation = (shoulder_pos[1] - joint_pos[1]) / person_height  # Note: Y increases downward
            metrics['shoulder_elevation'] = elevation
        else:
            metrics['shoulder_elevation'] = 0.0
            
        return metrics

    @staticmethod
    def create_fixed_windows(session_start_time_s: float, last_time_s: float, window_size_seconds: float, sliding_window_length_seconds: float):
        """
        Create fixed window boundaries for a session (in seconds).
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

    def extract_timestamps_and_keypoints(self, video_path: str, yolo_model_name: str = "yolo11m-pose", yolo_cache_dir: str = "./model_cache", output_fps: float = 8.0, debug: bool = False):
        """
        Extract timestamps and pose keypoints from video in a single pass to optimize memory usage.
        
        Args:
            video_path: Path to the video file
            yolo_model_name: Name of the YOLO pose model to use
            yolo_cache_dir: Directory containing the YOLO model cache
            output_fps: Target frame rate for output (default: 8.0). If video has higher fps, frames will be dropped.
            debug: Whether to show debug visualization
            
        Returns:
            Tuple of (frame_timestamps, keypoints_per_frame)
        """
        # Parse timestamps from filename for total duration
        start_time, end_time = self.parse_filename_timestamps(video_path)
        total_duration = (end_time - start_time).total_seconds()
        
        # Initialize YOLO pose estimator
        yolo_estimator = YoloPoseEstimator(model_name=yolo_model_name, cache_dir=yolo_cache_dir)
        
        # Process video frame by frame
        frame_timestamps = []
        keypoints_per_frame = []
        cap = None
        pbar = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame dropping parameters
            if video_fps <= 0:
                # Fallback: assume 30 fps if we can't get the actual fps
                raise ValueError(f"Could not determine video FPS, assuming {video_fps}")
            
            # Calculate how many frames to skip to achieve target output_fps
            if output_fps >= video_fps:
                # No need to drop frames, process all frames
                frame_skip = 1
                target_frames = total_frames
            else:
                # Calculate frame skip to achieve target fps
                frame_skip = max(1, int(video_fps / output_fps))
                target_frames = total_frames // frame_skip
                print(f"Video FPS: {video_fps:.1f}, Target FPS: {output_fps:.1f}, Skipping {frame_skip-1} out of every {frame_skip} frames")
            
            pbar = tqdm.tqdm(total=target_frames, desc="Extracting pose keypoints")
            
            frame_count = 0
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames to achieve target output_fps
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue

                # read first half of the frame if it is a depth video(i.e. 1280x480x3)
                if frame.shape[1] == 1280 and frame.shape[0] == 480:
                    frame = frame[:, :640, :] 
                    
                # Calculate timestamp for this frame
                if total_frames <= 1:
                    frame_time = start_time.timestamp()
                else:
                    frame_interval = total_duration / (total_frames - 1)
                    frame_time = start_time.timestamp() + frame_count * frame_interval
                
                # Extract pose keypoints
                keypoints_list, _, _, _ = yolo_estimator.process_frame(frame)
                if debug:
                    # show preview of pose viz
                    pose_viz = yolo_estimator.visualize_poses(frame, keypoints_list, show_metrics=True)
                    cv2.imshow("Pose Viz (Debug-Preview)", pose_viz)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if keypoints_list and (len(keypoints_list[0]) > 0) and (keypoints_list[0].shape[1] == 17):
                    keypoints_per_frame.append(keypoints_list[0])  # shape: (17, 3)
                else:
                    keypoints_per_frame.append(np.zeros((1, 17, 3)))  # No detection
                
                frame_timestamps.append(frame_time)
                processed_count += 1
                pbar.update(1)
                frame_count += 1
                
            if not frame_timestamps:
                raise ValueError(f"No frames could be processed from video: {video_path}")
            
            print(f"Processed {processed_count} frames out of {total_frames} total frames (effective FPS: {processed_count/total_duration:.1f})")
            return frame_timestamps, keypoints_per_frame
            
        except Exception as e:
            raise e
        
        finally:
            if pbar is not None:
                pbar.close()
            if cap is not None:
                cap.release()

    def generate_timestamp_windows(self, video_path: str, windows: list, yolo_model_name: str = "yolo11m-pose", yolo_cache_dir: str = "./model_cache", output_fps: float = 8.0, debug: bool = False):
        """
        For each window, select frames whose timestamp falls within window, extract pose features, and return window timestamps and features.
        Args:
            video_path: Path to MP4 video file
            windows: List of (win_start, win_end) tuples in seconds (relative to start)
            yolo_model_name: Name of the YOLO pose model to use
            yolo_cache_dir: Directory containing the YOLO model cache
            output_fps: Target frame rate for output (default: 8.0). If video has higher fps, frames will be dropped.
            debug: Whether to show debug visualization
        Returns:
            Tuple of (window_timestamps, feature_list) - one entry per window. If a window has no frames, None is appended for that window.
        """
        # Extract timestamps and keypoints in a single pass
        frame_timestamps, keypoints_per_frame = self.extract_timestamps_and_keypoints(video_path, yolo_model_name, yolo_cache_dir, output_fps, debug=debug)
        
        
        # For each window, select frames whose timestamp falls within window
        window_timestamps = []
        feature_list = []
        for win_start, win_end in tqdm.tqdm(windows, desc="Extracting pose features"):
            indices = [i for i, t in enumerate(frame_timestamps) if (t >= win_start and t < win_end)]
            if len(indices) > 0:
                window = [keypoints_per_frame[i] for i in indices]
                window_ts = win_end # use the end of the window as the timestamp for the features
                try:
                    features = self.extract_features(window)
                    window_timestamps.append(window_ts)
                    feature_list.append(features)
                except Exception as e:
                    print(f"Error extracting features for window {win_start}-{win_end}: {e}")
                    continue
        return window_timestamps, feature_list

    def generate_raw_windows(self, video_path: str, windows: list, session_output_dir: str, selected_windows: list, camera_name: str,
                                yolo_model_name: str = "yolo11m-pose", yolo_cache_dir: str = "./model_cache",
                                output_fps: float = 8.0, debug: bool = False):
        """
        Streaming pose extraction that only processes frames for selected windows.
        Args:
            video_path: Path to MP4 video file
            windows: List of (win_start, win_end) tuples in seconds (relative to start)
            selected_windows: List of window end times to process
            camera_name: Camera identifier for output files
            yolo_model_name: Name of the YOLO pose model to use
            yolo_cache_dir: Directory containing the YOLO model cache
            output_fps: Target frame rate for output (default: 8.0). If video has higher fps, frames will be dropped.
            debug: Whether to show debug visualization
        Returns:
            None
        """
        # create the session output dir
        os.makedirs(session_output_dir, exist_ok=True)

        # Parse timestamps from filename for total duration
        start_time, end_time = self.parse_filename_timestamps(video_path)
        total_duration = (end_time - start_time).total_seconds()
        
        # Initialize YOLO pose estimator
        yolo_estimator = YoloPoseEstimator(model_name=yolo_model_name, cache_dir=yolo_cache_dir)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if video_fps <= 0:
            cap.release()
            raise ValueError(f"Could not determine video FPS, got {video_fps}")
        
        # Calculate frame dropping parameters
        if output_fps >= video_fps:
            frame_skip = 1
        else:
            frame_skip = max(1, int(video_fps / output_fps))
            
        # Streaming logic
        window_idx = 0
        num_windows = len(windows)
        frame_buffer = []
        frame_buffer_times = []
        frame_count = 0
        
        pbar = tqdm.tqdm(total=total_frames, desc="Streaming raw frames from video")
        
        while True:
            ret, frame = cap.read()
            if not ret or window_idx >= num_windows:
                break
                
            pbar.update(1)
            
            # Skip frames to achieve target output_fps
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
                
            # Handle concatenated videos (read first half if it's a depth video)
            if frame.shape[1] == 1280 and frame.shape[0] == 480:
                frame = frame[:, :640, :]
                
            # Calculate timestamp for this frame
            if total_frames <= 1:
                frame_time = start_time.timestamp()
            else:
                frame_interval = total_duration / (total_frames - 1)
                frame_time = start_time.timestamp() + frame_count * frame_interval
                
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
                    
                    # Extract pose keypoints only for selected windows
                    window_keypoints = []
                    for i in indices:
                        raw_frame = frame_buffer[i]
                        keypoints_list, _, _, _ = yolo_estimator.process_frame(raw_frame)
                        
                        if debug:
                            pose_viz = yolo_estimator.visualize_poses(raw_frame, keypoints_list, show_metrics=True)
                            cv2.imshow("Pose Viz (Debug-Preview)", pose_viz)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        
                        if keypoints_list and (len(keypoints_list[0]) > 0) and (keypoints_list[0].shape[1] == 17):
                            window_keypoints.append(keypoints_list[0])  # shape: (17, 3)
                        else:
                            window_keypoints.append(np.zeros((1, 17, 3)))  # No detection
                    
                    instance_data = [(ts, pose) for ts, pose in zip(window_frame_timestamps, window_keypoints)]
                    if len(instance_data) > 0:
                        window_instance_dir = f"{session_output_dir}/{win_end}"
                        os.makedirs(window_instance_dir, exist_ok=True)
                        # save the raw pose windows as pickle file
                        window_pickle_path = f"{window_instance_dir}/pose_{camera_name}.pkl"
                        with open(window_pickle_path, "wb") as f:
                            pickle.dump(instance_data, f)
                        # save the pose visualization
                        window_viz_path = f"{window_instance_dir}/pose_{camera_name}.mp4"
                        cv2_writer = cv2.VideoWriter(window_viz_path, cv2.VideoWriter_fourcc(*'avc1'), output_fps, (640, 480))
                        for keypoints in window_keypoints:
                            frame_viz = yolo_estimator.visualize_raw_pose((480, 640), [keypoints])
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
                    
                    # Extract pose keypoints for remaining windows
                    window_keypoints = []
                    for i in indices:
                        raw_frame = frame_buffer[i]
                        keypoints_list, _, _, _ = yolo_estimator.process_frame(raw_frame)
                        
                        if keypoints_list and (len(keypoints_list[0]) > 0) and (keypoints_list[0].shape[1] == 17):
                            window_keypoints.append(keypoints_list[0])  # shape: (17, 3)
                        else:
                            window_keypoints.append(np.zeros((1, 17, 3)))  # No detection
                    
                    instance_data = [(ts, pose) for ts, pose in zip(window_frame_timestamps, window_keypoints)]
                    window_instance_dir = f"{session_output_dir}/{win_end}"
                    os.makedirs(window_instance_dir, exist_ok=True)
                    # save the raw pose windows as pickle file
                    window_pickle_path = f"{window_instance_dir}/pose_{camera_name}.pkl"
                    with open(window_pickle_path, "wb") as f:
                        pickle.dump(instance_data, f)
                    # save the pose visualization
                    window_viz_path = f"{window_instance_dir}/pose_{camera_name}.mp4"
                    cv2_writer = cv2.VideoWriter(window_viz_path, cv2.VideoWriter_fourcc(*'avc1'), output_fps, (640, 480))
                    for keypoints in window_keypoints:
                        frame_viz = yolo_estimator.visualize_raw_pose((480, 640), [keypoints])
                        cv2_writer.write(frame_viz)
                    cv2_writer.release()
            window_idx += 1
        
        # Release the YOLO estimator
        del yolo_estimator
        return None


    def visualize_features(self, window: List[np.ndarray], features: dict, show_window=True):
        """Visualize extracted pose features for debug purposes"""
        if len(window) == 0:
            return None
        
        # Create diagnostic info for header
        frame_count = len(window)
        valid_poses = sum(1 for frame in window if frame.shape[0] > 0 and np.any(frame[:, :, 2] > PoseConstants.MIN_CONFIDENCE))
        hands_active = features.get('hands_mean_velocity', 0) > 5.0
        high_zone_time = features.get('time_in_high_zone', 0)
        
        # Create figure with pose frame and overlaid features (horizontal layout)
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle(f'Pose Features | {frame_count} frames | Valid poses: {valid_poses} | Hands active: {"✓" if hands_active else "✗"} | High zone: {high_zone_time:.1%}', 
                     fontsize=14, fontweight='bold')
        
        # Left: Pose skeleton with overlays
        ax1 = plt.subplot(1, 2, 1)
        self._plot_pose_skeleton_with_overlays(ax1, window[-1])  # Use last frame
        
        # Right: Feature values overlaid horizontally
        ax2 = plt.subplot(1, 2, 2)
        self._plot_horizontal_pose_features(ax2, features)
        
        plt.tight_layout()
        
        if show_window:
            # Convert matplotlib figure to OpenCV image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
            
            # Convert RGBA to BGR for OpenCV (drop alpha channel)
            img = cv2.cvtColor(buf[:,:,:3], cv2.COLOR_RGB2BGR)
            
            # Display using OpenCV
            cv2.imshow('Pose Feature Visualization', img)
            cv2.waitKey(1)
            
        plt.close(fig)
        return fig
    
    def _plot_pose_skeleton_with_overlays(self, ax, frame):
        """Plot pose skeleton with feature overlays"""
        # Create blank image for pose visualization
        img_height, img_width = 480, 640  # Standard dimensions
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Handle frame structure: could be (1, 17, 3) or (17, 3)
        if len(frame.shape) == 3 and frame.shape[0] == 0:
            ax.imshow(img)
            ax.set_title('No Pose Detected')
            ax.axis('off')
            return
        elif len(frame.shape) == 3:
            pose = frame[0]  # Use first detected person (1, 17, 3) -> (17, 3)
        else:
            pose = frame  # Already (17, 3)
            
        # Check if any joints have sufficient confidence
        valid_joints = np.sum(pose[:, 2] > PoseConstants.MIN_CONFIDENCE)
        
        if valid_joints == 0:
            # No valid pose detected - create a message image
            cv2.putText(img, 'No Pose Detected', (img_width//4, img_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f'All {len(pose)} joints have confidence 0', (img_width//6, img_height//2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title('No Pose Detected - Check Input Data')
            ax.axis('off')
            return
        
        # Draw skeleton connections
        connections = [
            (PoseConstants.L_SHOULDER, PoseConstants.R_SHOULDER),  # Shoulders
            (PoseConstants.L_SHOULDER, PoseConstants.L_ELBOW),     # Left arm
            (PoseConstants.L_ELBOW, PoseConstants.L_WRIST),
            (PoseConstants.R_SHOULDER, PoseConstants.R_ELBOW),     # Right arm
            (PoseConstants.R_ELBOW, PoseConstants.R_WRIST),
            (PoseConstants.L_SHOULDER, PoseConstants.L_HIP),       # Left torso
            (PoseConstants.R_SHOULDER, PoseConstants.R_HIP),       # Right torso
            (PoseConstants.L_HIP, PoseConstants.R_HIP),            # Hips
        ]
        
        # Draw connections only for confident joints
        for joint1, joint2 in connections:
            if (pose[joint1][2] > PoseConstants.MIN_CONFIDENCE and 
                pose[joint2][2] > PoseConstants.MIN_CONFIDENCE):
                # Additional check that coordinates are reasonable (not at origin)
                pt1 = (int(pose[joint1][0]), int(pose[joint1][1]))
                pt2 = (int(pose[joint2][0]), int(pose[joint2][1]))
                
                # Skip if either point is at or near origin (likely invalid detection)
                if (pt1[0] > 5 and pt1[1] > 5 and pt2[0] > 5 and pt2[1] > 5 and
                    pt1[0] < img_width-5 and pt1[1] < img_height-5 and 
                    pt2[0] < img_width-5 and pt2[1] < img_height-5):
                    cv2.line(img, pt1, pt2, (100, 100, 100), 2)
        
        # Draw keypoints with color coding
        joint_colors = {
            'hands': [(PoseConstants.L_WRIST, PoseConstants.R_WRIST), (0, 255, 0)],  # Green
            'elbows': [(PoseConstants.L_ELBOW, PoseConstants.R_ELBOW), (255, 0, 0)], # Red
            'shoulders': [(PoseConstants.L_SHOULDER, PoseConstants.R_SHOULDER), (0, 0, 255)], # Blue
            'head': [(PoseConstants.NOSE,), (255, 255, 0)], # Yellow
        }
        
        for joint_type, (joints, color) in joint_colors.items():
            for joint_idx in joints:
                if pose[joint_idx][2] > PoseConstants.MIN_CONFIDENCE:
                    center = (int(pose[joint_idx][0]), int(pose[joint_idx][1]))
                    
                    # Only draw if coordinates are reasonable (not at origin or edge)
                    if (center[0] > 5 and center[1] > 5 and 
                        center[0] < img_width-5 and center[1] < img_height-5):
                        cv2.circle(img, center, 8, color, -1)
                        cv2.circle(img, center, 10, (255, 255, 255), 2)
        
        # Add working zone indicators only if hips are confidently detected and in valid positions
        if (pose[PoseConstants.L_HIP][2] > PoseConstants.MIN_CONFIDENCE and 
            pose[PoseConstants.R_HIP][2] > PoseConstants.MIN_CONFIDENCE):
            left_hip = (pose[PoseConstants.L_HIP][0], pose[PoseConstants.L_HIP][1])
            right_hip = (pose[PoseConstants.R_HIP][0], pose[PoseConstants.R_HIP][1])
            
            # Check if hip coordinates are reasonable (not at origin)
            if (left_hip[0] > 5 and left_hip[1] > 5 and right_hip[0] > 5 and right_hip[1] > 5 and
                left_hip[0] < img_width-5 and left_hip[1] < img_height-5 and 
                right_hip[0] < img_width-5 and right_hip[1] < img_height-5):
                
                # Calculate person height for relative zones
                # Try to use head as top reference, otherwise use shoulders
                head_pos = None
                if pose[PoseConstants.NOSE][2] > PoseConstants.MIN_CONFIDENCE:
                    head_pos = (pose[PoseConstants.NOSE][0], pose[PoseConstants.NOSE][1])
                    
                shoulder_pos = None
                valid_shoulders = []
                if pose[PoseConstants.L_SHOULDER][2] > PoseConstants.MIN_CONFIDENCE:
                    valid_shoulders.append((pose[PoseConstants.L_SHOULDER][0], pose[PoseConstants.L_SHOULDER][1]))
                if pose[PoseConstants.R_SHOULDER][2] > PoseConstants.MIN_CONFIDENCE:
                    valid_shoulders.append((pose[PoseConstants.R_SHOULDER][0], pose[PoseConstants.R_SHOULDER][1]))
                    
                if valid_shoulders:
                    shoulder_pos = (np.mean([s[0] for s in valid_shoulders]), np.mean([s[1] for s in valid_shoulders]))
                
                hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                top_ref = head_pos if head_pos is not None else shoulder_pos
                
                if top_ref is not None:
                    person_height = abs(hip_center[1] - top_ref[1])
                    
                    # Draw working zone boundaries using new relative thresholds
                    zone_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # Red, Green, Blue
                    zone_names = ['High', 'Mid', 'Low']
                    zone_thresholds = [0.3, 0.8, 1.5]  # Corresponding to ZONE_THRESHOLDS
                    
                    for i, (threshold, color, name) in enumerate(zip(zone_thresholds, zone_colors, zone_names)):
                        y_pos = int(top_ref[1] + threshold * person_height)
                        if y_pos > 0 and y_pos < img_height:  # Only draw if within image bounds
                            cv2.line(img, (0, y_pos), (img_width, y_pos), color, 2)
                            cv2.putText(img, f'{name} Zone', (10, y_pos - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('Pose Skeleton\nGreen=Hands, Red=Elbows, Blue=Shoulders')
        ax.axis('off')
    
    def _plot_horizontal_pose_features(self, ax, features):
        """Plot pose feature values as horizontal layout with compact groups"""
        # Check if all features are zero (indicating no valid pose data)
        non_zero_features = sum(1 for v in features.values() if abs(v) > 1e-6)
        
        if non_zero_features == 0:
            # All features are zero - show message
            ax.text(0.5, 0.5, 'No Valid Pose Data\nAll Features = 0', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=16, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Feature Values - No Data')
            ax.axis('off')
            return
        
        # Select most relevant features for kitchen activities
        feature_groups = {
            'Hand Activity': [
                'hands_mean_velocity', 'hands_max_velocity', 'right_hand_overhead_time', 'left_hand_overhead_time'
            ],
            'Reach & Elevation': [
                'right_hand_max_elevation', 'left_hand_max_elevation', 'combined_lateral_range', 'mean_lateral_activity'
            ],
            'Working Zones': [
                'time_in_high_zone', 'time_in_mid_zone', 'time_in_low_zone', 'right_hand_height_range'
            ],
            'Body Posture': [
                'torso_tilt_mean', 'right_elbow_mean', 'left_elbow_mean', 'pose_symmetry_mean'
            ]
        }
        
        # Create 4 horizontal sections for each group
        colors = ['lightgreen', 'skyblue', 'lightcoral', 'plum']
        y_positions = [0.8, 0.6, 0.4, 0.2]  # 4 horizontal bands
        
        for i, (group_name, feature_list) in enumerate(feature_groups.items()):
            y_pos = y_positions[i]
            
            # Group header
            ax.text(0.02, y_pos + 0.08, group_name, fontweight='bold', fontsize=10, 
                   transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
            
            # Feature values in horizontal list
            x_start = 0.02
            x_spacing = 0.23  # Space between features horizontally
            
            valid_features = [f for f in feature_list if f in features]
            
            for j, feature_name in enumerate(valid_features[:4]):  # Limit to 4 features per row
                value = features.get(feature_name, 0)
                
                # Create short display name
                short_name = feature_name.replace('_mean', '').replace('_velocity', '_vel')\
                    .replace('hands_', 'h_').replace('right_', 'r_').replace('left_', 'l_')\
                    .replace('time_in_', '').replace('_zone', '').replace('workspace_area', 'workspace')\
                    .replace('torso_tilt', 'torso').replace('elbow', 'elb').replace('pose_', '')\
                    .replace('bilateral_', 'bil_').replace('_ratio', '')
                
                x_pos = x_start + (j * x_spacing)
                
                # Display feature name and value
                ax.text(x_pos, y_pos + 0.03, short_name, fontsize=8, fontweight='bold', 
                       transform=ax.transAxes)
                ax.text(x_pos, y_pos - 0.02, f'{value:.3f}', fontsize=8, 
                       color=colors[i], transform=ax.transAxes)
                
                # Small bar indicator
                bar_width = 0.15
                bar_height = 0.02
                
                # Normalize values for display based on feature type
                if 'velocity' in feature_name:
                    norm_value = min(value / 50.0, 1.0)  # Scale for hand velocities
                elif 'zone' in feature_name or feature_name.endswith('_ratio'):
                    norm_value = min(value, 1.0)  # Already normalized (0-1)
                elif 'workspace' in feature_name:
                    norm_value = min(value / 10000.0, 1.0)  # Scale for workspace area
                elif 'range' in feature_name:
                    norm_value = min(value / 500.0, 1.0)  # Scale for horizontal range
                elif 'angle' in feature_name or 'elbow' in feature_name:
                    norm_value = min(abs(value) / 180.0, 1.0)  # Scale for angles
                else:
                    norm_value = min(abs(value), 1.0)  # Default scaling
                
                # Draw small horizontal bar
                bar_x = x_pos
                bar_y = y_pos - 0.06
                ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width * norm_value, bar_height, 
                                         facecolor=colors[i], alpha=0.7, transform=ax.transAxes))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Pose Feature Values', fontsize=12, pad=10)
        ax.axis('off')  # Clean look without axes


if __name__ == "__main__":
    extractor = PoseFeatures()
    # video_path = "/Users/prasoon/Research/VAX/datasets/autonomous_phase3/phase3_processed/P5data-collection/processed_video_data/P11-00-20250404_102012_20250404_102402.mp4"
    video_path = "/Users/prasoon/Research/VAX/Datasets/autonomous_phase3/phase3_processed/P2-data-collection/processed_video_data/P2-22-20250414_055644_20250414_064953.mp4"
    yolo_model_name = "yolo11n-pose"
    yolo_cache_dir = "/Users/prasoon/Research/VAX/OrganicHAR/models/pose"
    output_fps = 8.0  # Target output frame rate
    
    start_time, end_time = extractor.parse_filename_timestamps(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_timestamps_dt = extractor.calculate_frame_timestamps(start_time, end_time, total_frames)
    frame_timestamps = [ts.timestamp() for ts in frame_timestamps_dt]
    last_time = frame_timestamps[-1]
    session_start_time_s = frame_timestamps[0]
    windows = extractor.create_fixed_windows(session_start_time_s, last_time, window_size_seconds=5.0, sliding_window_length_seconds=0.5)
    timestamps, features = extractor.generate_timestamp_windows(video_path, windows, yolo_model_name, yolo_cache_dir, output_fps)
    # Convert to numpy array
    feature_names = list(features[0].keys())
    features_array = np.array([[features.get(name, 0.) for name in feature_names] 
                                for features in features])
    print(f"Processed {len(timestamps)} windows")
    print(f"Time range: {timestamps[0]} to {timestamps[-1]}")
    print(f"Feature names ({len(features[0])} total): {list(features[0].keys())}")
    print(f"First window features: {features[0]}")
    print(f"Features array shape: {features_array.shape}")
