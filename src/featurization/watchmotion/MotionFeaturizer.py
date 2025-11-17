import os
import numpy as np
import pandas as pd
import time
import pickle
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
import warnings
from tqdm import tqdm
import glob
warnings.filterwarnings("ignore")


class MotionFeaturizer:
    def __init__(self, cache_dir="./feature_cache"):
        """
        Initialize the Motion Featurizer with traditional signal processing
        
        Args:
            cache_dir (str): Directory to cache outputs
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Feature names for the 36-feature vector
        self.feature_names = [
            # Accelerometer magnitude features (1-3)
            'accel_mag_mean', 'accel_mag_std', 'accel_mag_max',
            # Accelerometer per-axis std (4-6)
            'accel_std_x', 'accel_std_y', 'accel_std_z',
            # Accelerometer correlations (7-9)
            'accel_corr_xy', 'accel_corr_xz', 'accel_corr_yz',
            # Signal magnitude area (10)
            'signal_magnitude_area',
            # Tilt angle (11)
            'tilt_angle_mean',
            # Gyroscope magnitude features (12-14)
            'gyro_mag_mean', 'gyro_mag_std', 'gyro_mag_max',
            # Gyroscope per-axis std (15-17)
            'gyro_std_x', 'gyro_std_y', 'gyro_std_z',
            # Cross-sensor features (18-20)
            'accel_gyro_corr', 'energy_ratio', 'combined_intensity',
            # Frequency features (21-23)
            'dominant_freq', 'spectral_centroid', 'spectral_entropy',
            # Temporal features (24-26)
            'autocorr_0_5s', 'zero_crossing_rate', 'step_count',
            # Energy bands (27-30)
            'energy_band_0_0_3hz', 'energy_band_0_3_3hz', 'energy_band_3_15hz', 'periodicity_strength',
            # Magnetometer features (31-33)
            'mag_mean', 'mag_std', 'heading_change',
            # Advanced features (34-36)
            'movement_smoothness', 'activity_intensity', 'postural_transitions'
        ]
        
        # Initialize FPS tracking
        self.fps_history = []
        self.max_fps_history = 30
        
        print(f"Motion Featurizer initialized with {len(self.feature_names)} features")
        
    def load_motion_data(self, base_path, watch_type="android"):
        """
        Load motion data from separate accelerometer, gyroscope, and magnetometer files
        
        Args:
            base_path (str): Base path for motion data files (without extension)
            
        Returns:
            dict: Dictionary containing aligned motion data
        """
        if watch_type == "android":
            # Load the three sensor files
            acc_file = f"{base_path}_acc.txt"
            gyro_file = f"{base_path}_gyro.txt"
            mag_file = f"{base_path}_mag.txt"
            
            # Read data files
            acc_data = pd.read_csv(acc_file, names=['local_timestamp', 'epoch_timestamp', 'x', 'y', 'z']).sort_values(by='epoch_timestamp')
            gyro_data = pd.read_csv(gyro_file, names=['local_timestamp', 'epoch_timestamp', 'x', 'y', 'z']).sort_values(by='epoch_timestamp')
            mag_data = pd.read_csv(mag_file, names=['local_timestamp', 'epoch_timestamp', 'x', 'y', 'z']).sort_values(by='epoch_timestamp')
            
            # Convert timestamps to common time base (epoch_timestamp is in milliseconds)
            # acc_data['time'] = (acc_data['epoch_timestamp'] - acc_data['epoch_timestamp'].min()) / 1000.0
            # gyro_data['time'] = (gyro_data['epoch_timestamp'] - gyro_data['epoch_timestamp'].min()) / 1000.0
            # mag_data['time'] = (mag_data['epoch_timestamp'] - mag_data['epoch_timestamp'].min()) / 1000.0
            acc_data['epoch_timestamp'] = acc_data['epoch_timestamp'] / 1e3
            gyro_data['epoch_timestamp'] = gyro_data['epoch_timestamp'] / 1e3
            mag_data['epoch_timestamp'] = mag_data['epoch_timestamp'] / 1e3

        elif watch_type == "iwatch":
            # Load the three sensor files
            motion_files = glob.glob(f"{base_path}*motion*.txt")
            acc_data, gyro_data, mag_data = [], [], []
            for motion_file in tqdm(motion_files, desc="Loading motion files"):
                motion_columns = [
                    'epoch_timestamp',
                    'accel_x', 'accel_y', 'accel_z',
                    'gravity_x', 'gravity_y', 'gravity_z',
                    'rot_x', 'rot_y', 'rot_z',
                    'mag_x', 'mag_y', 'mag_z',
                    'roll', 'pitch', 'unnamed','yaw',
                    'quat_x', 'quat_y', 'quat_z', 'quat_w', 'local_ts_diff'
                ]
                df_motion_data = pd.read_csv(motion_file, names=motion_columns,sep=' ',index_col=False)
                acc_data.append(df_motion_data[['epoch_timestamp', 'accel_x', 'accel_y', 'accel_z']])
                gyro_data.append(df_motion_data[['epoch_timestamp', 'rot_x', 'rot_y', 'rot_z']])
                mag_data.append(df_motion_data[['epoch_timestamp', 'mag_x', 'mag_y', 'mag_z']])
            acc_data = pd.concat(acc_data, axis=0).sort_values(by='epoch_timestamp').rename(columns={'accel_x': 'x', 'accel_y': 'y', 'accel_z': 'z'})
            gyro_data = pd.concat(gyro_data, axis=0).sort_values(by='epoch_timestamp').rename(columns={'rot_x': 'x', 'rot_y': 'y', 'rot_z': 'z'})
            mag_data = pd.concat(mag_data, axis=0).sort_values(by='epoch_timestamp').rename(columns={'mag_x': 'x', 'mag_y': 'y', 'mag_z': 'z'})
            
        return {
            'accelerometer': acc_data[['epoch_timestamp', 'x', 'y', 'z']].values,
            'gyroscope': gyro_data[['epoch_timestamp', 'x', 'y', 'z']].values,
            'magnetometer': mag_data[['epoch_timestamp', 'x', 'y', 'z']].values
        }
    
    def create_sliding_windows(self, motion_data, window_size=2.0, slide_length=0.2):
        """
        Create sliding windows from motion data
        
        Args:
            motion_data (dict): Dictionary containing motion sensor data
            window_size (float): Window size in seconds
            slide_length (float): Sliding window step size in seconds
            
        Returns:
            list: List of windowed motion data segments
        """
        windows = []
        
        # Find the common time range across all sensors
        min_time = max([data[:, 0].min() for data in motion_data.values()])
        max_time = min([data[:, 0].max() for data in motion_data.values()])
        
        # Generate window start times
        window_start = min_time
        
        while window_start + window_size <= max_time:
            window_end = window_start + window_size
            
            # Extract data for this window from each sensor
            window_data = {}
            for sensor_name, sensor_data in motion_data.items():
                # Find indices for this time window
                mask = (sensor_data[:, 0] >= window_start) & (sensor_data[:, 0] < window_end)
                window_data[sensor_name] = sensor_data[mask]
            
            # Only include windows with sufficient data
            if all(len(data) > 10 for data in window_data.values()):
                windows.append({
                    'start_time': window_start,
                    'end_time': window_end,
                    'data': window_data
                })
            
            window_start += slide_length
        
        return windows
    
    def interpolate_to_fixed_rate(self, sensor_data, target_rate=50.0):
        """
        Interpolate sensor data to a fixed sampling rate
        
        Args:
            sensor_data (np.ndarray): Sensor data with time in first column
            target_rate (float): Target sampling rate in Hz
            
        Returns:
            np.ndarray: Interpolated data at fixed rate
        """
        if len(sensor_data) < 2:
            return sensor_data
        
        # Extract time and data columns
        time_orig = sensor_data[:, 0]
        data_orig = sensor_data[:, 1:4]  # x, y, z columns
        
        # Create new time vector at target rate
        time_start = time_orig[0]
        time_end = time_orig[-1]
        time_new = np.arange(time_start, time_end, 1.0/target_rate)
        
        # Interpolate each axis
        data_new = np.zeros((len(time_new), 3))
        for i in range(3):
            data_new[:, i] = np.interp(time_new, time_orig, data_orig[:, i])
        
        # Combine time and data
        return np.column_stack([time_new, data_new])
    
    def extract_features_from_window(self, window_data, target_rate=50.0):
        """
        Extract traditional signal processing features from windowed motion data
        
        Args:
            window_data (dict): Dictionary containing windowed sensor data
            target_rate (float): Target sampling rate in Hz
            
        Returns:
            np.ndarray: Feature vector of length 36
        """
        # Interpolate all sensors to fixed rate
        interpolated_data = {}
        for sensor_name, sensor_data in window_data.items():
            interpolated_data[sensor_name] = self.interpolate_to_fixed_rate(sensor_data, target_rate)
        
        # Get sensor data
        acc_data = interpolated_data['accelerometer']
        gyro_data = interpolated_data['gyroscope']
        mag_data = interpolated_data['magnetometer']
        
        # Ensure all have same length (trim to shortest)
        min_len = min(len(acc_data), len(gyro_data), len(mag_data))
        acc_data = acc_data[:min_len]
        gyro_data = gyro_data[:min_len]
        mag_data = mag_data[:min_len]
        
        # Extract xyz data (skip time column)
        acc_xyz = acc_data[:, 1:4]
        gyro_xyz = gyro_data[:, 1:4]
        mag_xyz = mag_data[:, 1:4]
        
        # Calculate magnitude vectors
        acc_mag = np.sqrt(np.sum(acc_xyz**2, axis=1))
        gyro_mag = np.sqrt(np.sum(gyro_xyz**2, axis=1))
        mag_mag = np.sqrt(np.sum(mag_xyz**2, axis=1))
        
        # Initialize features array
        features = np.zeros(36)
        
        # 1-3. Accelerometer magnitude features
        features[0] = np.mean(acc_mag)
        features[1] = np.std(acc_mag)
        features[2] = np.max(acc_mag)
        
        # 4-6. Accelerometer per-axis std
        features[3] = np.std(acc_xyz[:, 0])  # x
        features[4] = np.std(acc_xyz[:, 1])  # y
        features[5] = np.std(acc_xyz[:, 2])  # z
        
        # 7-9. Accelerometer correlations
        features[6] = np.corrcoef(acc_xyz[:, 0], acc_xyz[:, 1])[0, 1] if len(acc_xyz) > 1 else 0
        features[7] = np.corrcoef(acc_xyz[:, 0], acc_xyz[:, 2])[0, 1] if len(acc_xyz) > 1 else 0
        features[8] = np.corrcoef(acc_xyz[:, 1], acc_xyz[:, 2])[0, 1] if len(acc_xyz) > 1 else 0
        
        # 10. Signal magnitude area (SMA)
        features[9] = np.sum(np.abs(acc_xyz[:, 0]) + np.abs(acc_xyz[:, 1]) + np.abs(acc_xyz[:, 2]))
        
        # 11. Tilt angle mean
        features[10] = np.mean(np.arctan2(acc_xyz[:, 2], np.sqrt(acc_xyz[:, 0]**2 + acc_xyz[:, 1]**2)))
        
        # 12-14. Gyroscope magnitude features
        features[11] = np.mean(gyro_mag)
        features[12] = np.std(gyro_mag)
        features[13] = np.max(gyro_mag)
        
        # 15-17. Gyroscope per-axis std
        features[14] = np.std(gyro_xyz[:, 0])  # x
        features[15] = np.std(gyro_xyz[:, 1])  # y
        features[16] = np.std(gyro_xyz[:, 2])  # z
        
        # 18-20. Cross-sensor features
        features[17] = np.corrcoef(acc_mag, gyro_mag)[0, 1] if len(acc_mag) > 1 else 0
        gyro_energy = np.sum(gyro_mag**2)
        acc_energy = np.sum(acc_mag**2)
        features[18] = gyro_energy / (acc_energy + 1e-8)  # energy ratio
        features[19] = np.sqrt(np.mean(acc_mag**2) + np.mean(gyro_mag**2))  # combined intensity
        
        # 21-23. Frequency features (using accelerometer magnitude)
        if len(acc_mag) > 4:
            # FFT of accelerometer magnitude
            acc_fft = np.abs(fft(acc_mag))
            freqs = fftfreq(len(acc_mag), 1/target_rate)
            
            # Only use positive frequencies
            n_half = len(acc_fft) // 2
            acc_fft = acc_fft[:n_half]
            freqs = freqs[:n_half]
            
            # Dominant frequency
            dominant_idx = np.argmax(acc_fft[1:]) + 1  # Skip DC component
            features[20] = freqs[dominant_idx] if dominant_idx < len(freqs) else 0
            
            # Spectral centroid
            features[21] = np.sum(freqs * acc_fft) / (np.sum(acc_fft) + 1e-8)
            
            # Spectral entropy
            psd_norm = acc_fft / (np.sum(acc_fft) + 1e-8)
            features[22] = entropy(psd_norm + 1e-8)
        else:
            features[20:23] = 0
        
        # 24-26. Temporal features
        # Autocorrelation at 0.5s lag
        lag_samples = int(0.5 * target_rate)
        if len(acc_mag) > lag_samples:
            features[23] = np.corrcoef(acc_mag[:-lag_samples], acc_mag[lag_samples:])[0, 1]
        else:
            features[23] = 0
        
        # Zero crossing rate
        zero_crossings = np.sum((acc_mag[:-1] * acc_mag[1:]) < 0)
        features[24] = zero_crossings / (len(acc_mag) - 1) if len(acc_mag) > 1 else 0
        
        # Step count (peaks in filtered acceleration)
        filtered_acc = signal.butter(4, [0.5, 3], btype='band', fs=target_rate, output='sos')
        if len(acc_mag) > 8:  # Need minimum length for filtering
            acc_filtered = signal.sosfilt(filtered_acc, acc_mag)
            peaks, _ = signal.find_peaks(acc_filtered, height=np.std(acc_filtered), distance=int(0.3*target_rate))
            features[25] = len(peaks)
        else:
            features[25] = 0
        
        # 27-30. Energy bands
        if len(acc_mag) > 4:
            # Calculate power spectral density
            freqs, psd = signal.welch(acc_mag, fs=target_rate, nperseg=min(len(acc_mag), 64))
            
            # Energy in different frequency bands
            band_0_0_3 = np.sum(psd[(freqs >= 0) & (freqs <= 0.3)])
            band_0_3_3 = np.sum(psd[(freqs > 0.3) & (freqs <= 3)])
            band_3_15 = np.sum(psd[(freqs > 3) & (freqs <= 15)])
            total_energy = np.sum(psd) + 1e-8
            
            features[26] = band_0_0_3 / total_energy
            features[27] = band_0_3_3 / total_energy
            features[28] = band_3_15 / total_energy
            
            # Periodicity strength (energy in human movement frequency range)
            human_band = np.sum(psd[(freqs >= 0.3) & (freqs <= 3.5)])
            features[29] = human_band / total_energy
        else:
            features[26:30] = 0
        
        # 31-33. Magnetometer features
        features[30] = np.mean(mag_mag)
        features[31] = np.std(mag_mag)
        
        # Heading change (compass direction change)
        if len(mag_xyz) > 1:
            heading_start = np.arctan2(mag_xyz[0, 1], mag_xyz[0, 0])
            heading_end = np.arctan2(mag_xyz[-1, 1], mag_xyz[-1, 0])
            heading_change = np.abs(heading_end - heading_start)
            features[32] = min(heading_change, 2*np.pi - heading_change)  # Shortest angular distance
        else:
            features[32] = 0
        
        # 34-36. Advanced features
        # Movement smoothness (negative sum of squared jerk)
        if len(acc_mag) > 2:
            jerk = np.diff(np.diff(acc_mag))  # Second derivative
            features[33] = -np.sum(jerk**2)
        else:
            features[33] = 0
        
        # Activity intensity score
        features[34] = np.log(1 + np.var(acc_mag) + np.var(gyro_mag))
        
        # Postural transitions (large changes in tilt angle)
        if len(acc_xyz) > 1:
            tilt_angles = np.arctan2(acc_xyz[:, 2], np.sqrt(acc_xyz[:, 0]**2 + acc_xyz[:, 1]**2))
            tilt_changes = np.abs(np.diff(tilt_angles))
            features[35] = np.sum(tilt_changes > 0.2)  # Count significant tilt changes
        else:
            features[35] = 0
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features

    def process_window(self, window_data):
        """
        Process a single window of motion data to extract features
        
        Args:
            window_data (dict): Dictionary containing windowed sensor data
            
        Returns:
            tuple: (features_vector, feature_breakdown, processing_time)
        """
        start_time = time.time()
        
        # Extract features
        features = self.extract_features_from_window(window_data)
        
        # Create feature breakdown for visualization/analysis
        feature_breakdown = dict(zip(self.feature_names, features))
        
        process_time = time.time() - start_time
        
        # Update FPS tracking
        self.fps_history.append(1.0 / process_time if process_time > 0 else 0)
        if len(self.fps_history) > self.max_fps_history:
            self.fps_history.pop(0)
        
        return features, feature_breakdown, process_time
    
    def get_average_fps(self):
        """Get the average FPS from recent processing"""
        if len(self.fps_history) == 0:
            return 0.0
        return np.mean(self.fps_history)
    
    def setup_live_visualization(self):
        """
        Setup live visualization for features
        """
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle('Live Motion Feature Visualization')
        
        # Setup subplots
        self.axes[0, 0].set_title('Accelerometer Features')
        self.axes[0, 1].set_title('Gyroscope Features')
        self.axes[1, 0].set_title('Cross-Sensor & Frequency Features')
        self.axes[1, 1].set_title('Advanced Features')
        
        plt.tight_layout()
        plt.show(block=False)
        
        # Initialize feature tracking
        self.feature_history = []
        self.timestamps = []
        self.max_history = 100  # Keep last 100 windows
    
    def update_live_visualization(self, features, feature_breakdown, window_info):
        """
        Update live visualization with new feature data

        Args:
            features (np.ndarray): Feature vector
            feature_breakdown (dict): Named features
            window_info (dict): Window metadata
        """
        if not hasattr(self, 'fig'):
            return
            
        # Add to history
        self.feature_history.append(features)
        self.timestamps.append(window_info.get('start_time', 0))
        
        # Keep only recent history
        if len(self.feature_history) > self.max_history:
            self.feature_history.pop(0)
            self.timestamps.pop(0)
        
        if len(self.feature_history) < 2:
            return
        
        # Convert to numpy array for easier indexing
        feature_array = np.array(self.feature_history)
        time_array = np.array(self.timestamps)
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot accelerometer features (0-10)
        self.axes[0, 0].plot(time_array, feature_array[:, 0], 'r-', label='Acc Mag Mean')
        self.axes[0, 0].plot(time_array, feature_array[:, 1], 'g-', label='Acc Mag Std')
        self.axes[0, 0].plot(time_array, feature_array[:, 9]/100, 'b-', label='SMA/100')
        self.axes[0, 0].set_title('Accelerometer Features')
        self.axes[0, 0].set_xlabel('Time (s)')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True)
        
        # Plot gyroscope features (11-16)
        self.axes[0, 1].plot(time_array, feature_array[:, 11], 'r-', label='Gyro Mag Mean')
        self.axes[0, 1].plot(time_array, feature_array[:, 12], 'g-', label='Gyro Mag Std')
        self.axes[0, 1].plot(time_array, feature_array[:, 19], 'b-', label='Combined Intensity')
        self.axes[0, 1].set_title('Gyroscope Features')
        self.axes[0, 1].set_xlabel('Time (s)')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True)
        
        # Plot cross-sensor and frequency features (17-29)
        self.axes[1, 0].plot(time_array, feature_array[:, 20], 'r-', label='Dominant Freq')
        self.axes[1, 0].plot(time_array, feature_array[:, 21], 'g-', label='Spectral Centroid')
        self.axes[1, 0].plot(time_array, feature_array[:, 25], 'b-', label='Step Count')
        self.axes[1, 0].set_title('Cross-Sensor & Frequency Features')
        self.axes[1, 0].set_xlabel('Time (s)')
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True)
        
        # Plot advanced features (30-35)
        self.axes[1, 1].plot(time_array, feature_array[:, 34], 'r-', label='Activity Intensity')
        self.axes[1, 1].plot(time_array, feature_array[:, 35], 'g-', label='Postural Transitions')
        self.axes[1, 1].plot(time_array, feature_array[:, 30], 'b-', label='Mag Mean')
        self.axes[1, 1].set_title('Advanced Features')
        self.axes[1, 1].set_xlabel('Time (s)')
        self.axes[1, 1].legend()
        self.axes[1, 1].grid(True)
        
        # Update title with window information
        live_hz = window_info.get("live_hz", None)
        hz_text = f' | Live: {live_hz} Hz' if live_hz else ''
        self.fig.suptitle(f'Live Motion Features - Window {window_info.get("window_index", 0)+1} | '
                         f'Data Time: {window_info.get("start_time", 0):.2f}s | '
                         f'Processing FPS: {window_info.get("avg_fps", 0):.1f}{hz_text}')
        
        # Refresh the display
        plt.draw()
        plt.pause(0.001)  # Small pause to allow GUI to update
    
    def close_live_visualization(self):
        """
        Close the live visualization
        """
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            plt.ioff()  # Turn off interactive mode
    
    def process_motion_files(self, base_path, window_size=2.0, slide_length=0.2, 
                           output_path=None, show_preview=False, live_hz=None, watch_type="android"):
        """
        Process motion files to extract features using sliding windows
        
        Args:
            base_path (str): Base path for motion data files
            window_size (float): Window size in seconds
            slide_length (float): Sliding window step size in seconds
            output_path (str, optional): Path for output data file
            show_preview (bool): Whether to show live feature visualization
            live_hz (float, optional): Hz rate for live simulation (e.g., 50 Hz)
            
        Returns:
            tuple: (features_list, metadata)
        """
        # Load motion data
        print(f"Loading motion data from {base_path}...")
        motion_data = self.load_motion_data(base_path, watch_type=watch_type)
        
        # Create sliding windows
        print(f"Creating sliding windows (window_size={window_size}s, slide_length={slide_length}s)...")
        windows = self.create_sliding_windows(motion_data, window_size, slide_length)
        print(f"Created {len(windows)} windows")
        
        # Setup live visualization if requested
        if show_preview:
            self.setup_live_visualization()
            if live_hz:
                print(f"Live feature visualization started at {live_hz} Hz. Close the window to stop.")
            else:
                print("Live feature visualization started. Close the window to stop.")
        
        # Process each window
        features_list = []
        feature_breakdowns = []
        metadata = []
        
        start_time = time.time()
        live_start_time = None
        
        try:
            for i, window in tqdm(enumerate(windows), total=len(windows), desc="Processing windows"):
                # Calculate live timing if live_hz is specified
                if live_hz and show_preview:
                    if live_start_time is None:
                        live_start_time = time.time()
                        first_window_time = window['start_time']
                    
                    # Calculate when this window should be processed based on actual data timestamps
                    data_elapsed_time = window['start_time'] - first_window_time
                    expected_time = data_elapsed_time / live_hz
                    elapsed_time = time.time() - live_start_time
                    
                    # Sleep to maintain the specified Hz rate
                    if elapsed_time < expected_time:
                        sleep_time = expected_time - elapsed_time
                        time.sleep(sleep_time)
                        print(f"  Waiting {sleep_time:.3f}s to maintain {live_hz} Hz rate...")
                
                # Process this window
                features, feature_breakdown, process_time = self.process_window(window['data'])
                
                # Store results
                features_list.append(features)
                feature_breakdowns.append(feature_breakdown)
                window_metadata = {
                    'window_index': i,
                    'start_time': window['start_time'],
                    'end_time': window['end_time'],
                    'process_time': process_time,
                    'avg_fps': self.get_average_fps(),
                    'live_hz': live_hz
                }
                metadata.append(window_metadata)
                
                # Update live visualization if requested
                if show_preview:
                    self.update_live_visualization(features, feature_breakdown, window_metadata)
                    
                    # Check if window is still open
                    if not plt.fignum_exists(self.fig.number):
                        print("Visualization window closed. Stopping processing.")
                        break
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        finally:
            # Close live visualization
            if show_preview:
                self.close_live_visualization()
        
        # Generate output path if not provided
        if output_path is None:
            base_name = Path(base_path).stem
            output_path = f"{base_name}_motion_features.pickle"
        
        # Save results
        results = {
            'features': features_list,
            'feature_breakdowns': feature_breakdowns,
            'feature_names': self.feature_names,
            'metadata': metadata,
            'parameters': {
                'window_size': window_size,
                'slide_length': slide_length,
                'feature_count': len(self.feature_names)
            }
        }
        
        print(f"Saving results to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        total_time = time.time() - start_time
        print(f"\nProcessing complete:")
        print(f"Processed {len(windows)} windows in {total_time:.2f} seconds")
        print(f"Average FPS: {self.get_average_fps():.2f}")
        print(f"Features saved to: {output_path}")
        print(f"Feature vector shape: ({len(self.feature_names)},)")
        
        return features_list, metadata
    
    def generate_timestamp_windows(self, base_path, windows, watch_type="android", debug=False, delta_seconds=0):
        """
        Generate timestamp windows for a android watch motion data file
        """
        # Load motion data
        print(f"Loading motion data from {base_path}...")
        motion_data = self.load_motion_data(base_path, watch_type=watch_type)
        accel_data = motion_data['accelerometer']
        gyro_data = motion_data['gyroscope']
        mag_data = motion_data['magnetometer']
        accel_data[:,0] += delta_seconds
        gyro_data[:,0] += delta_seconds
        mag_data[:,0] += delta_seconds
        
        # process each window and get the features
        timestamps = []
        features = []
        for window in tqdm(windows, desc="Processing motion windows", leave=False):
            window_start, window_end = window
            window_data = {
                    'accelerometer': accel_data[(accel_data[:,0] >= window_start) & (accel_data[:,0] < window_end)],
                    'gyroscope': gyro_data[(gyro_data[:,0] >= window_start) & (gyro_data[:,0] < window_end)],
                    'magnetometer': mag_data[(mag_data[:,0] >= window_start) & (mag_data[:,0] < window_end)]
            }
            if (len(window_data['accelerometer']) == 0) or (len(window_data['gyroscope']) == 0) or (len(window_data['magnetometer']) == 0):
                continue
            window_features, _, _ = self.process_window(window_data)
            timestamps.append(window_end)
            features.append(window_features)
            
        return timestamps, features
    
    def generate_raw_windows(self, base_path, windows, session_output_dir: str, selected_windows: list, watch_hand_lr: str, watch_type="android", debug=False, delta_seconds=0):
        """
        Generate raw motion data for selected windows
        """
        # Load motion data
        print(f"Loading motion data from {base_path}...")
        motion_data = self.load_motion_data(base_path, watch_type=watch_type)
        accel_data = motion_data['accelerometer']
        gyro_data = motion_data['gyroscope']
        mag_data = motion_data['magnetometer']
        accel_data[:,0] += delta_seconds
        gyro_data[:,0] += delta_seconds
        mag_data[:,0] += delta_seconds
        
        # Process each window and save raw data for selected windows
        for window in tqdm(windows, desc="Processing motion windows for raw data", leave=False):
            window_start, window_end = window
            if window_end in selected_windows:
                # Create window instance directory
                window_instance_dir = f"{session_output_dir}/{window_end}"
                os.makedirs(window_instance_dir, exist_ok=True)

                # separate files for accelerometer, gyroscope and magnetometer
                accel_data_path = f"{window_instance_dir}/{watch_hand_lr}_accel.csv"
                gyro_data_path = f"{window_instance_dir}/{watch_hand_lr}_gyro.csv"
                mag_data_path = f"{window_instance_dir}/{watch_hand_lr}_mag.csv"
                df_accel = pd.DataFrame(accel_data[(accel_data[:,0] >= window_start) & (accel_data[:,0] < window_end)])
                df_gyro = pd.DataFrame(gyro_data[(gyro_data[:,0] >= window_start) & (gyro_data[:,0] < window_end)])
                df_mag = pd.DataFrame(mag_data[(mag_data[:,0] >= window_start) & (mag_data[:,0] < window_end)])
                if df_accel.shape[0] > 0:
                    df_accel.to_csv(accel_data_path, index=False, header=False)
                if df_gyro.shape[0] > 0:
                    df_gyro.to_csv(gyro_data_path, index=False, header=False)
                if df_mag.shape[0] > 0:
                    df_mag.to_csv(mag_data_path, index=False, header=False)

        return None
            
    
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


def main():
    parser = argparse.ArgumentParser(description="Motion Feature Extractor - Traditional Signal Processing")
    parser.add_argument("--cache-dir", type=str, default="./feature_cache",
                       help="Directory to cache outputs")
    parser.add_argument("--data", type=str, 
                       default="sample_data/watch-dominant_1743776412909",
                       help="Base path to motion data files (without _acc.txt suffix)")
    parser.add_argument("--window-size", type=float, default=2.0,
                       help="Window size in seconds (default: 2.0)")
    parser.add_argument("--slide-length", type=float, default=0.2,
                       help="Sliding window step size in seconds (default: 0.2)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (defaults to auto-generated)")
    parser.add_argument("--show-preview", action="store_true",
                       help="Show live feature visualization during processing")
    parser.add_argument("--live-hz", type=float, default=None,
                       help="Hz rate for live simulation (e.g., 5.0 for 5 Hz)")
    parser.add_argument("--watch-type", type=str, default="android",
                       help="Watch type (android or iwatch)")
    
    args = parser.parse_args()
    args.data = "/Volumes/Research-Prasoon/OrganicHAR/Datasets/autonomous_phase3/phase3_processed/P4--data-collection/sessions/P4-06/watch-dominant"
    args.watch_type = "iwatch"
    # Initialize motion featurizer
    featurizer = MotionFeaturizer(
        cache_dir=args.cache_dir
    )
    
    # Process motion files
    features, metadata = featurizer.process_motion_files(
        base_path=args.data,
        window_size=args.window_size,
        slide_length=args.slide_length,
        output_path=args.output,
        show_preview=args.show_preview,
        live_hz=args.live_hz,
        watch_type=args.watch_type
    )
    
    print(f"\nExtracted {len(features)} feature vectors")
    if len(features) > 0:
        print(f"Feature vector shape: {features[0].shape}")
        print(f"Feature names: {featurizer.feature_names}")
        
        # Show sample features from first window
        print(f"\nSample features from first window:")
        for i, (name, value) in enumerate(zip(featurizer.feature_names, features[0])):
            print(f"  {i+1:2d}. {name:25s}: {value:8.4f}")


if __name__ == "__main__":
    main()