import os
import cv2
import numpy as np
import time
import pickle
from pathlib import Path
from ultralytics import YOLO
import subprocess
from ultralytics.utils.downloads import attempt_download_asset


class YoloPoseEstimator:
    def __init__(self, model_name="yolo11m-pose", cache_dir="./model_cache", force_export=False):
        """
        Initialize the YOLO Pose Estimator with CoreML support for Apple Metal

        Args:
            model_name (str): Name of the YOLO model to use (default: yolo11m-pose)
            cache_dir (str): Directory to cache models
            force_export (bool): Force re-export of CoreML model even if it exists
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.short_model_name = model_name.split('/')[-1]  # Extract just the model name without path

        # For pose models, we need to specify a variant without NMS
        self.coreml_model_path = self.cache_dir / f"{self.short_model_name}_640x640_no_nms.mlpackage"

        # Download the model to cache_dir if not present, and load from there
        model_file = self.cache_dir / (self.short_model_name if self.short_model_name.endswith('.pt') else f"{self.short_model_name}.pt")
        if not model_file.exists():
            print(f"Downloading {self.model_name} to {model_file} ...")
            attempt_download_asset(self.model_name, dir=self.cache_dir)
        self.pt_model = YOLO(str(model_file))

        # Export and load the CoreML model if possible
        self.coreml_model = self._export_and_load_coreml(force_export)

        # Define keypoint connections (skeleton)
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]

        # Define keypoint names (COCO format)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        # Define keypoint colors for visualization
        self.keypoint_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                                (0, 255, 255), (255, 0, 255), (128, 0, 0),
                                (0, 128, 0), (0, 0, 128), (128, 128, 0),
                                (0, 128, 128), (128, 0, 128), (64, 0, 0),
                                (0, 64, 0), (0, 0, 64), (64, 64, 0), (0, 64, 64)]

        # Define skeleton colors
        self.skeleton_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                                (0, 255, 255), (255, 0, 255), (128, 0, 0),
                                (0, 128, 0), (0, 0, 128), (128, 128, 0),
                                (0, 128, 128), (128, 0, 128), (64, 0, 0),
                                (0, 64, 0), (0, 0, 64), (64, 64, 0), (0, 64, 64),
                                (64, 0, 64), (128, 128, 128)]

        # Track frame rates for smooth display
        self.fps_history = []

    def _export_and_load_coreml(self, force_export):
        """
        Export the model to CoreML format with special settings for pose models

        Args:
            force_export (bool): Force re-export even if model exists

        Returns:
            YOLO model or None
        """
        if not self.coreml_model_path.exists() or force_export:
            print(f"Exporting {self.model_name} to CoreML format...")

            try:
                # For pose models, explicitly disable NMS
                # Use the model file from cache_dir for export
                model_file = self.cache_dir / (self.short_model_name if self.short_model_name.endswith('.pt') else f"{self.short_model_name}.pt")
                export_cmd = [
                    "yolo", "export",
                    f"model={str(model_file)}",
                    "format=coreml",
                    "imgsz=640",
                    "half=True",
                    "nms=False"  # Critical for pose models
                ]

                # Run the export command
                subprocess.run(export_cmd, check=True)

                # Move exported model to our cache directory with correct name
                exported_model = Path(f"{self.cache_dir}/{self.short_model_name}.mlpackage")
                if exported_model.exists():
                    if self.coreml_model_path.exists():
                        os.remove(self.coreml_model_path)
                    os.rename(exported_model, self.coreml_model_path)
                    print(f"Model exported and cached at {self.coreml_model_path}")
                else:
                    raise FileNotFoundError(f"Expected exported model at {exported_model} not found")

            except Exception as e:
                print(f"Error during CoreML export: {e}")
                return None
        else:
            print(f"Using cached CoreML model from {self.coreml_model_path}")

        try:
            # Load the CoreML model
            coreml_model = YOLO(str(self.coreml_model_path))
            print("Successfully loaded CoreML model")
            return coreml_model
        except Exception as e:
            print(f"Error loading CoreML model: {e}")
            return None

    def process_frame(self, frame):
        """
        Process a single video frame to detect poses

        Args:
            frame (numpy.ndarray): Input video frame (BGR format, 640x480)

        Returns:
            tuple: (list of detected keypoints per person, padded frame, used_coreml, fps)
        """
        # Create a square frame by padding with black borders
        orig_h, orig_w = frame.shape[:2]
        target_size = 640

        # Create a square black canvas
        square_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        # Copy the original frame onto the square canvas at the top
        square_frame[0:orig_h, 0:orig_w] = frame

        # Track which model we used and performance
        used_coreml = False
        fps = 0

        # Try CoreML inference if available
        if self.coreml_model is not None:
            try:
                start_time = time.time()

                # Run inference with CoreML model
                coreml_results = self.coreml_model.predict(square_frame, verbose=False)
                inference_time = time.time() - start_time

                # Check if we got valid results with keypoints
                if any(hasattr(r, 'keypoints') and r.keypoints is not None for r in coreml_results):
                    used_coreml = True
                    results = coreml_results
                    fps = 1.0 / inference_time
                else:
                    results = None
            except Exception:
                results = None
        else:
            results = None

        # Fall back to PyTorch if CoreML failed or didn't detect keypoints
        if results is None:
            start_time = time.time()
            results = self.pt_model.predict(square_frame, verbose=False)
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time

        # Extract keypoints
        all_keypoints = []

        for result in results:
            # Get keypoints from the result
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                # Get keypoints array, shape (num_people, num_keypoints, 3)
                # Where each keypoint is [x, y, confidence]
                keypoints = result.keypoints.data.cpu().numpy()
                all_keypoints.append(keypoints)

        # Update FPS history for smoother display
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:  # Keep last 30 frames
            self.fps_history.pop(0)

        return all_keypoints, square_frame, used_coreml, fps

    def calculate_pose_metrics(self, keypoints_batch):
        """
        Calculate additional pose metrics for visualization

        Args:
            keypoints_batch: Batch of keypoint data for one or more people

        Returns:
            dict: Dictionary of pose metrics
        """
        metrics = {}

        # Handle empty batch case
        if keypoints_batch is None or len(keypoints_batch) == 0:
            return metrics

        # Additional check for empty keypoints array
        if keypoints_batch.size == 0 or keypoints_batch.shape[0] == 0:
            return metrics

        # Now we can safely get the first person's keypoints
        try:
            person_keypoints = keypoints_batch[0]

            # Check if person_keypoints is valid and has the expected shape
            if person_keypoints.size == 0 or person_keypoints.shape[0] == 0:
                return metrics

            # Calculate overall pose confidence
            valid_keypoints = person_keypoints[:, 2] > 0.5
            if np.any(valid_keypoints):
                metrics['overall_confidence'] = np.mean(person_keypoints[valid_keypoints, 2])
            else:
                metrics['overall_confidence'] = 0.0

            # Check if key body parts are detected
            key_parts = {
                'face': [0, 1, 2, 3, 4],  # nose, eyes, ears
                'upper_body': [5, 6, 7, 8, 9, 10],  # shoulders, elbows, wrists
                'lower_body': [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
            }

            for part_name, indices in key_parts.items():
                # Make sure indices are within bounds
                valid_indices = [i for i in indices if i < person_keypoints.shape[0]]
                if not valid_indices:
                    metrics[f'{part_name}_confidence'] = 0.0
                    continue

                part_keypoints = person_keypoints[valid_indices]
                valid_part_keypoints = part_keypoints[:, 2] > 0.5
                if np.any(valid_part_keypoints):
                    metrics[f'{part_name}_confidence'] = np.mean(part_keypoints[valid_part_keypoints, 2])
                else:
                    metrics[f'{part_name}_confidence'] = 0.0

            # Calculate body angle (if shoulders and hips detected)
            # Make sure all indices are within bounds first
            left_shoulder_idx, right_shoulder_idx = 5, 6
            left_hip_idx, right_hip_idx = 11, 12

            max_idx = person_keypoints.shape[0] - 1
            if (left_shoulder_idx <= max_idx and right_shoulder_idx <= max_idx and
                    left_hip_idx <= max_idx and right_hip_idx <= max_idx and
                    person_keypoints[left_shoulder_idx, 2] > 0.5 and
                    person_keypoints[right_shoulder_idx, 2] > 0.5 and
                    person_keypoints[left_hip_idx, 2] > 0.5 and
                    person_keypoints[right_hip_idx, 2] > 0.5):
                # Get midpoints
                shoulder_mid_x = (person_keypoints[left_shoulder_idx, 0] + person_keypoints[right_shoulder_idx, 0]) / 2
                shoulder_mid_y = (person_keypoints[left_shoulder_idx, 1] + person_keypoints[right_shoulder_idx, 1]) / 2

                hip_mid_x = (person_keypoints[left_hip_idx, 0] + person_keypoints[right_hip_idx, 0]) / 2
                hip_mid_y = (person_keypoints[left_hip_idx, 1] + person_keypoints[right_hip_idx, 1]) / 2

                # Calculate angle from vertical
                dx = shoulder_mid_x - hip_mid_x
                dy = shoulder_mid_y - hip_mid_y
                body_angle = np.degrees(np.arctan2(dx, -dy))  # negative dy because y-axis is inverted
                metrics['body_angle'] = body_angle

        except Exception as e:
            # If anything goes wrong during metrics calculation, log it and return empty metrics
            print(f"Error calculating pose metrics: {e}")
            return {}

        return metrics

    def visualize_poses(self, frame, keypoints, show_metrics=True, thickness=2, circle_radius=4):
        """
        Visualize detected poses on the frame with enhanced information

        Args:
            frame (numpy.ndarray): Input video frame (square padded frame)
            keypoints (list): List of keypoints from process_frame
            show_metrics (bool): Whether to show additional pose metrics
            thickness (int): Line thickness for skeleton
            circle_radius (int): Radius of keypoint circles

        Returns:
            numpy.ndarray: Frame with visualized poses
        """
        # Create a copy of the frame
        output_frame = frame.copy()

        # Get the frame dimensions
        frame_h, frame_w = output_frame.shape[:2]
        orig_h = 480  # Original height before padding

        # Count people detected
        num_people = 0
        for batch in keypoints:
            if batch is not None and hasattr(batch, 'shape'):
                num_people += batch.shape[0]

        # Draw dividing line between video and metrics area
        cv2.line(output_frame, (0, orig_h), (frame_w, orig_h), (128, 128, 128), 1)

        # Add summary info in the black area
        cv2.putText(output_frame, f"People detected: {num_people}",
                    (10, orig_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # For each keypoints batch in the list
        person_counter = 0
        for person_idx, person_keypoints_batch in enumerate(keypoints):
            # Skip if the batch is None or empty
            if person_keypoints_batch is None or not hasattr(person_keypoints_batch,
                                                             'shape') or person_keypoints_batch.size == 0:
                continue

            # Calculate additional metrics for this person (safely)
            metrics = self.calculate_pose_metrics(person_keypoints_batch)

            # For each person's keypoints
            for person_subidx in range(person_keypoints_batch.shape[0]):
                # Skip if out of bounds
                if person_subidx >= person_keypoints_batch.shape[0]:
                    continue

                # Get this person's keypoints
                try:
                    person_keypoints = person_keypoints_batch[person_subidx]
                except IndexError:
                    continue  # Skip if index is invalid

                # Current person index across all batches
                current_person_idx = person_counter
                person_counter += 1

                # Make sure person_keypoints is valid
                if person_keypoints is None or not hasattr(person_keypoints, 'shape') or person_keypoints.size == 0:
                    continue

                # Draw each keypoint
                for i in range(person_keypoints.shape[0]):
                    try:
                        x, y, conf = person_keypoints[i]
                        if conf > 0.5:  # Confidence threshold
                            color = self.keypoint_colors[i % len(self.keypoint_colors)]
                            cv2.circle(output_frame, (int(x), int(y)), circle_radius, color, -1)

                            # Add keypoint labels for important points
                            if i in [0, 5, 6, 11, 12, 15, 16] and i < len(
                                    self.keypoint_names):  # nose, shoulders, hips, ankles
                                label = self.keypoint_names[i]
                                cv2.putText(output_frame, f"{label}", (int(x) + 5, int(y) - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    except (IndexError, ValueError):
                        continue  # Skip invalid keypoints

                # Draw skeleton
                for i, (start_idx, end_idx) in enumerate(self.skeleton):
                    try:
                        if (start_idx < person_keypoints.shape[0] and
                                end_idx < person_keypoints.shape[0] and
                                person_keypoints[start_idx, 2] > 0.5 and
                                person_keypoints[end_idx, 2] > 0.5):
                            color = self.skeleton_colors[i % len(self.skeleton_colors)]
                            start_point = (int(person_keypoints[start_idx, 0]), int(person_keypoints[start_idx, 1]))
                            end_point = (int(person_keypoints[end_idx, 0]), int(person_keypoints[end_idx, 1]))

                            cv2.line(output_frame, start_point, end_point, color, thickness)
                    except (IndexError, ValueError):
                        continue  # Skip invalid connections

                # Show person ID
                try:
                    if person_keypoints.shape[0] > 0 and np.any(person_keypoints[:, 2] > 0.5):
                        # Find the highest keypoint with good confidence as anchor for person ID
                        valid_indices = np.where(person_keypoints[:, 2] > 0.5)[0]
                        if len(valid_indices) > 0:  # Check that we have valid keypoints
                            highest_y = np.min(person_keypoints[valid_indices, 1])
                            highest_idx = valid_indices[np.argmin(person_keypoints[valid_indices, 1])]
                            id_x = int(person_keypoints[highest_idx, 0])
                            id_y = int(highest_y) - 10

                            cv2.putText(output_frame, f"Person {current_person_idx + 1}",
                                        (id_x, id_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                except (IndexError, ValueError):
                    continue  # Skip if can't find valid keypoint for ID

                # Display metrics for this person if requested in black area below original frame
                if show_metrics and metrics:
                    # Calculate position based on person index
                    metrics_width = 210  # Width of each metrics panel
                    metrics_height = 130  # Height of each metrics panel
                    metrics_per_row = frame_w // metrics_width  # Number of metrics panels per row

                    # Calculate position in the black area
                    panel_idx = current_person_idx % metrics_per_row  # Position in current row
                    row_idx = current_person_idx // metrics_per_row  # Row number

                    # Calculate panel position
                    metrics_x = 5 + (panel_idx * metrics_width)
                    metrics_y = orig_h + 30 + (row_idx * (metrics_height + 5))

                    # Background for metrics panel
                    cv2.rectangle(output_frame,
                                  (metrics_x, metrics_y),
                                  (metrics_x + metrics_width - 5, metrics_y + metrics_height),
                                  (0, 0, 0), -1)
                    cv2.rectangle(output_frame,
                                  (metrics_x, metrics_y),
                                  (metrics_x + metrics_width - 5, metrics_y + metrics_height),
                                  (255, 255, 255), 1)

                    # Title
                    cv2.putText(output_frame, f"Person {current_person_idx + 1}",
                                (metrics_x + 5, metrics_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Overall confidence
                    if 'overall_confidence' in metrics:
                        conf = metrics['overall_confidence'] * 100
                        cv2.putText(output_frame, f"Conf: {conf:.1f}%",
                                    (metrics_x + 5, metrics_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Body part confidences
                    if 'face_confidence' in metrics:
                        conf = metrics['face_confidence'] * 100
                        cv2.putText(output_frame, f"Face: {conf:.1f}%",
                                    (metrics_x + 5, metrics_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if 'upper_body_confidence' in metrics:
                        conf = metrics['upper_body_confidence'] * 100
                        cv2.putText(output_frame, f"Upper: {conf:.1f}%",
                                    (metrics_x + 5, metrics_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if 'lower_body_confidence' in metrics:
                        conf = metrics['lower_body_confidence'] * 100
                        cv2.putText(output_frame, f"Lower: {conf:.1f}%",
                                    (metrics_x + 5, metrics_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Body angle
                    if 'body_angle' in metrics:
                        angle = metrics['body_angle']
                        cv2.putText(output_frame, f"Angle: {angle:.1f}°",
                                    (metrics_x + 5, metrics_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output_frame
    
    def visualize_raw_pose(self, frame_size, keypoints_data, confidence_threshold=0.3):
        """
        Visualize pose data on a black background as a creative silhouette.

        This function creates a stylized representation of the pose, suitable for
        analysis by vision language models. It uses geometric shapes to approximate
        body parts, without any text labels.

        Args:
            frame_size (tuple): The (height, width) of the output image.
            keypoints_data (list): List of keypoint batches from process_frame.
            confidence_threshold (float): Minimum confidence to consider a keypoint.

        Returns:
            numpy.ndarray: A black image with the pose silhouette.
        """
        height, width = frame_size[:2]
        output_image = np.zeros((height, width, 3), dtype=np.uint8)

        # A standard skeleton definition for COCO 17 keypoints
        skeleton = [
            # Arms
            (5, 7), (7, 9), (6, 8), (8, 10),
            # Legs
            (11, 13), (13, 15), (12, 14), (14, 16),
            # Torso connections
            (5, 6), (11, 12), (5, 11), (6, 12)
        ]

        for person_keypoints_batch in keypoints_data:
            if person_keypoints_batch is None or not hasattr(person_keypoints_batch, 'shape') or person_keypoints_batch.size == 0:
                continue

            for person_idx in range(person_keypoints_batch.shape[0]):
                person_keypoints = person_keypoints_batch[person_idx]
                
                # Helper to check if a keypoint is valid (confident and not at origin)
                def is_valid(kpt_idx):
                    if kpt_idx >= len(person_keypoints):
                        return False
                    kpt = person_keypoints[kpt_idx]
                    return kpt[2] > confidence_threshold and (kpt[0] != 0 or kpt[1] != 0)

                # Ensure there are enough valid keypoints to draw a person
                valid_kpt_count = sum(1 for i in range(len(person_keypoints)) if is_valid(i))
                if valid_kpt_count < 5:
                    continue

                # --- Body Scale Estimation ---
                ls_idx, rs_idx = 5, 6
                if is_valid(ls_idx) and is_valid(rs_idx):
                    shoulder_width = np.linalg.norm(person_keypoints[ls_idx, :2] - person_keypoints[rs_idx, :2])
                    base_thickness = int(shoulder_width / 4)
                else:
                    base_thickness = 10
                
                base_thickness = max(5, min(base_thickness, 30))

                # --- Draw Limbs and Skeleton ---
                limb_color = (160, 160, 160)
                for start_idx, end_idx in skeleton:
                    if is_valid(start_idx) and is_valid(end_idx):
                        
                        start_pt = tuple(person_keypoints[start_idx, :2].astype(int))
                        end_pt = tuple(person_keypoints[end_idx, :2].astype(int))
                        
                        # Use slightly thinner lines for arms and legs
                        is_limb = (start_idx, end_idx) in [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
                        thickness = int(base_thickness * 0.75) if is_limb else base_thickness
                        
                        cv2.line(output_image, start_pt, end_pt, limb_color, thickness, cv2.LINE_AA)

                # --- Draw Torso ---
                torso_indices = [5, 6, 12, 11]  # L-shoulder, R-shoulder, R-hip, L-hip
                torso_points = [person_keypoints[i, :2] for i in torso_indices if is_valid(i)]
                
                if len(torso_points) >= 3:
                    torso_points = np.array(torso_points, dtype=np.int32)
                    hull = cv2.convexHull(torso_points)
                    cv2.drawContours(output_image, [hull], -1, (200, 200, 200), -1, cv2.LINE_AA)

                # --- Draw Head ---
                head_indices = [0, 1, 2, 3, 4]  # Nose, L-eye, R-eye, L-ear, R-ear
                head_points = [person_keypoints[i, :2] for i in head_indices if is_valid(i)]

                if len(head_points) >= 3:
                    head_points = np.array(head_points, dtype=np.int32)
                    try:
                        ellipse = cv2.fitEllipse(head_points)
                        # Make ellipse slightly larger to better represent a head
                        center, axes, angle = ellipse
                        axes = (axes[0] * 1.2, axes[1] * 1.5)
                        ellipse = (center, axes, angle)
                        cv2.ellipse(output_image, ellipse, (220, 220, 220), -1, cv2.LINE_AA)
                    except cv2.error:
                        # Fallback to a circle if ellipse fitting fails
                        center = np.mean(head_points, axis=0)
                        radius = np.max(np.linalg.norm(head_points - center, axis=1))
                        cv2.circle(output_image, tuple(center.astype(int)), int(radius * 1.2), (220, 220, 220), -1, cv2.LINE_AA)

                # --- Draw Facial Features ---
                le_idx, re_idx = 1, 2
                if is_valid(le_idx) and is_valid(re_idx):
                    left_eye = tuple(person_keypoints[le_idx, :2].astype(int))
                    right_eye = tuple(person_keypoints[re_idx, :2].astype(int))
                    eye_radius = max(2, int(base_thickness / 5))
                    cv2.circle(output_image, left_eye, eye_radius, (20, 20, 20), -1, cv2.LINE_AA)
                    cv2.circle(output_image, right_eye, eye_radius, (20, 20, 20), -1, cv2.LINE_AA)
        
        return output_image    

    def process_video(self, video_path, output_path=None, show_preview=True, raw_pose=False, output_fps=8):
        """
        Process a video file to detect poses in each frame

        Args:
            video_path (str): Path to the input video file
            output_path (str, optional): Path for output video, or None to use auto naming
            show_preview (bool): Whether to display preview during processing
            raw_pose (bool): Whether to visualize the raw pose
            output_fps (int): Output FPS for the pose data extraction.
        Returns:
            tuple: (output_video_path, keypoints_data_path)
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, int(fps / output_fps))
        proc_w, proc_h = 640, 480
        # Prepare output video writer (using square 640x640 output)
        if output_path is None:
            # Auto-generate output path with model name
            input_path = Path(video_path)
            model_suffix = self.short_model_name.replace('.pt', '')
            output_path = str(input_path.with_name(f"{input_path.stem}_{model_suffix}_pose{input_path.suffix}"))

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (640, 640))

        # Dictionary to store keypoints for all frames
        all_frames_keypoints = {}

        # Process frames
        frame_idx = 0
        start_time = time.time()
        coreml_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # add skip logic
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # if depth frame (1280x480), cut the width so that the aspect ratio is maintained, and then scale it to 640x480
            if frame.shape[1] == 1280 and frame.shape[0] == 480:
                frame = frame[:, 640:, :]
            elif frame.shape[1] != proc_w and frame.shape[0] != proc_h:
                # we wish to resize it to 640x480, first cut the width so that the aspect ratio is maintained, and then scale it to 640x480
                in_w, in_h = frame.shape[1], frame.shape[0]
                in_aspect_ratio = in_w / in_h
                proc_aspect_ratio = proc_w / proc_h
                if in_aspect_ratio > proc_aspect_ratio:
                    new_w = int(in_h * proc_aspect_ratio)
                    left_start = (in_w - new_w) // 2
                    right_end = left_start + new_w
                    frame = frame[:, left_start:right_end]
                else:
                    new_h = int(in_w / proc_aspect_ratio)
                    top_start = (in_h - new_h) // 2
                    bottom_end = top_start + new_h
                    frame = frame[top_start:bottom_end, :]
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            # Process the frame and get keypoints and padded square frame
            keypoints, square_frame, used_coreml, current_fps = self.process_frame(frame)

            # Store keypoints with frame number as key
            all_frames_keypoints[frame_idx] = keypoints

            if used_coreml:
                coreml_frames += 1

            # Calculate average FPS
            elapsed_time = time.time() - start_time
            avg_fps = (frame_idx + 1) / elapsed_time if elapsed_time > 0 else 0

            # Calculate smooth FPS (moving average)
            smooth_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

            # Visualize and write to output video
            if raw_pose:
                output_frame = self.visualize_raw_pose(square_frame.shape[:2], keypoints)
                # Concise FPS overlay
                cv2.putText(output_frame, f"FPS: {smooth_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
            else:
                output_frame = self.visualize_poses(square_frame, keypoints)

            # Add frame and FPS info
            backend = "CoreML" if used_coreml else "PyTorch"
            if not raw_pose:
                cv2.putText(output_frame, f"Frame: {frame_idx}/{total_frames}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(output_frame, f"FPS: {smooth_fps:.1f} ({backend})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Write to output video
            out.write(output_frame)

            # Show preview if requested
            if show_preview:
                cv2.imshow("Pose Estimation", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Add screenshot capability
                    screenshot_path = str(Path(output_path).with_name(f"{Path(output_path).stem}_frame{frame_idx}.jpg"))
                    cv2.imwrite(screenshot_path, output_frame)
                    print(f"Screenshot saved: {screenshot_path}")

            # Print progress (less frequently)
            frame_idx += 1
            # if frame_idx % 30 == 0:
            #     coreml_pct = (coreml_frames / frame_idx) * 100 if frame_idx > 0 else 0
            #     print(
            #         f"Processed {frame_idx}/{total_frames} frames - Avg FPS: {avg_fps:.1f} - CoreML: {coreml_pct:.1f}%")

            # After processing a few frames, if CoreML is failing, disable it to save time
            if frame_idx >= 10 and coreml_frames == 0 and self.coreml_model is not None:
                print("CoreML not working, switching to PyTorch only")
                self.coreml_model = None

        # Clean up
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        # Save keypoints to pickle file
        keypoints_path = str(Path(output_path).with_suffix('.pkl'))
        with open(keypoints_path, 'wb') as f:
            pickle.dump(all_frames_keypoints, f)

        # print(f"\nVideo processing complete:")
        # print(f"- Output video: {output_path}")
        # print(f"- Keypoints data: {keypoints_path}")

        return output_path, keypoints_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Pose Estimator - Final Production Version")
    parser.add_argument("--model", type=str, default="yolo11n-pose",
                        help="Model to use (default: yolo11m-pose)")
    parser.add_argument("--cache-dir", type=str, default="./model_cache",
                        help="Directory to cache models")
    parser.add_argument("--video", type=str, required=False,
                        help="Path to the input video file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for output video (defaults to input_modelname_pose.mp4)")
    parser.add_argument("--force-export", action="store_true",
                        help="Force re-export of CoreML model even if it exists")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable preview window during processing")

    args = parser.parse_args()
    # args.video="sample5.mp4"
    args.video="/Users/prasoon/Research/Autonomous/autonomous-v2-mac/data_preprocessing/pose_estimation/sample5.mp4"

    # Initialize pose estimator
    pose_estimator = YoloPoseEstimator(
        model_name=args.model,
        cache_dir=args.cache_dir,
        force_export=args.force_export
    )

    # Process video
    output_path, keypoints_path = pose_estimator.process_video(
        video_path=args.video,
        output_path=args.output,
        show_preview=not args.no_preview  # Preview enabled by default
    )

    print("Done!")


if __name__ == "__main__":
    main()