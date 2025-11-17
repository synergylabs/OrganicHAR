import os
import cv2
import numpy as np
import time
import pickle
from pathlib import Path
from ultralytics import YOLO
import subprocess
import traceback

class YoloSegmentation:
    def __init__(self, model_name="yolo11m-seg", cache_dir="./model_cache", force_export=False, debug=False):
        """
        Initialize the YOLO Segmentation with CoreML support for Apple Metal

        Args:
            model_name (str): Name of the YOLO segmentation model to use (default: yolo11m-seg)
            cache_dir (str): Directory to cache models
            force_export (bool): Force re-export of CoreML model even if it exists
            debug (bool): Whether to print debug information
        """
        self.debug = debug
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.short_model_name = model_name.split('/')[-1]  # Extract just the model name without path

        # For segmentation models
        self.coreml_model_path = self.cache_dir / f"{self.short_model_name}_640x640.mlpackage"

        # Load the PyTorch model (always needed for fallback)
        self.pt_model = YOLO(f"{self.cache_dir}/{self.short_model_name}.pt")

        # Export and load the CoreML model if possible
        self.coreml_model = self._export_and_load_coreml(force_export)

        # Define class colors for visualization (COCO classes)
        self.class_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (0, 128, 128), (128, 0, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
            (64, 64, 0), (0, 64, 64), (64, 0, 64), (128, 128, 128), (255, 128, 0),
            (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255),
            (255, 255, 128), (255, 128, 255), (128, 255, 255), (192, 192, 192), (128, 128, 255),
            (255, 128, 128), (128, 255, 128), (255, 192, 128), (255, 128, 192), (128, 255, 192),
            (192, 255, 128), (128, 192, 255), (192, 128, 255), (255, 192, 192), (192, 255, 192),
            (192, 192, 255), (0, 0, 0), (64, 64, 64), (128, 128, 128), (192, 192, 192),
            (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (0, 128, 128), (128, 0, 128), (64, 0, 0), (0, 64, 0),
            (0, 0, 64), (64, 64, 0), (0, 64, 64), (64, 0, 64), (255, 128, 0),
            (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255),
            (255, 255, 128), (255, 128, 255), (128, 255, 255), (192, 192, 192), (128, 128, 255),
            (255, 128, 128), (128, 255, 128), (255, 192, 128), (255, 128, 192), (128, 255, 192)
        ]

        # Track frame rates for smooth display
        self.fps_history = []

    def _export_and_load_coreml(self, force_export):
        """
        Export the model to CoreML format

        Args:
            force_export (bool): Force re-export even if model exists

        Returns:
            YOLO model or None
        """
        if not self.coreml_model_path.exists() or force_export:
            print(f"Exporting {self.model_name} to CoreML format...")

            try:
                # Export to CoreML
                export_cmd = [
                    "yolo", "export",
                    f"model={self.model_name}",
                    "format=coreml",
                    "imgsz=640",
                    "half=True"
                ]

                # Run the export command
                subprocess.run(export_cmd, check=True)

                # Move exported model to our cache directory with correct name
                exported_model = Path(f"{self.short_model_name}.mlpackage")
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
        Process a single video frame to detect and segment objects

        Args:
            frame (numpy.ndarray): Input video frame (BGR format, 640x480)

        Returns:
            tuple: (list of segmentation results, padded frame, used_coreml, fps)
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

                # Check if we got valid results with masks
                if any(hasattr(r, 'masks') and r.masks is not None for r in coreml_results):
                    used_coreml = True
                    results = coreml_results
                    fps = 1.0 / inference_time
                else:
                    results = None
            except Exception:
                results = None
                print(f"Error during CoreML inference for frame({frame.shape}).")
                print(traceback.format_exc())
        else:
            results = None

        # Fall back to PyTorch if CoreML failed or didn't detect masks
        if results is None:
            start_time = time.time()
            results = self.pt_model.predict(square_frame, verbose=False)
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time

        # Extract segmentation results
        all_segmentations = []

        for result in results:
            # Get masks and boxes from the result
            if hasattr(result, 'masks') and result.masks is not None:
                # Get segmentation data
                seg_data = {
                    'masks': result.masks.data.cpu().numpy() if result.masks.data is not None else None,
                    'boxes': result.boxes.data.cpu().numpy() if result.boxes is not None else None
                }
                all_segmentations.append(seg_data)

        # Update FPS history for smoother display
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:  # Keep last 30 frames
            self.fps_history.pop(0)

        return all_segmentations, square_frame, used_coreml, fps

    def calculate_segmentation_metrics(self, segmentations):
        """
        Calculate additional segmentation metrics for visualization

        Args:
            segmentations: Batch of segmentation data

        Returns:
            dict: Dictionary of segmentation metrics
        """
        metrics = {}

        # Handle empty segmentations case
        if not segmentations or len(segmentations) == 0:
            return metrics

        # Collect all masks and boxes
        all_masks = []
        all_boxes = []
        
        for seg_batch in segmentations:
            if seg_batch is not None:
                if seg_batch.get('masks') is not None:
                    masks = seg_batch['masks']
                    if masks.size > 0:
                        all_masks.extend(masks)
                
                if seg_batch.get('boxes') is not None:
                    boxes = seg_batch['boxes']
                    if boxes.size > 0:
                        all_boxes.extend(boxes)

        if not all_masks and not all_boxes:
            return metrics

        # Calculate metrics from boxes (contains confidence and class info)
        if all_boxes:
            all_boxes = np.array(all_boxes)
            
            # Calculate overall segmentation confidence
            if all_boxes.size > 0:
                confidences = all_boxes[:, 4]  # Confidence is the 5th column
                metrics['avg_confidence'] = np.mean(confidences)
                metrics['max_confidence'] = np.max(confidences)
                metrics['min_confidence'] = np.min(confidences)

            # Count segmentations by confidence threshold
            high_conf_segments = np.sum(all_boxes[:, 4] > 0.8)
            med_conf_segments = np.sum((all_boxes[:, 4] > 0.5) & (all_boxes[:, 4] <= 0.8))
            low_conf_segments = np.sum(all_boxes[:, 4] <= 0.5)

            metrics['high_conf_count'] = high_conf_segments
            metrics['med_conf_count'] = med_conf_segments
            metrics['low_conf_count'] = low_conf_segments
            metrics['total_segments'] = len(all_boxes)

        # Calculate mask-specific metrics
        if all_masks:
            all_masks = np.array(all_masks)
            
            # Calculate average mask area
            mask_areas = []
            for mask in all_masks:
                if mask.size > 0:
                    area = np.sum(mask)
                    mask_areas.append(area)
            
            if mask_areas:
                metrics['avg_mask_area'] = np.mean(mask_areas)
                metrics['max_mask_area'] = np.max(mask_areas)
                metrics['min_mask_area'] = np.min(mask_areas)

        return metrics

    def visualize_segmentations(self, frame, segmentations, show_metrics=True, alpha=0.6):
        """
        Visualize segmentation masks on the frame with enhanced information

        Args:
            frame (numpy.ndarray): Input video frame (square padded frame)
            segmentations (list): List of segmentation results from process_frame
            show_metrics (bool): Whether to show additional segmentation metrics
            alpha (float): Transparency of mask overlay (0.0 to 1.0)

        Returns:
            numpy.ndarray: Frame with visualized segmentations
        """
        # Create a copy of the frame
        output_frame = frame.copy()

        # Get the frame dimensions
        frame_h, frame_w = output_frame.shape[:2]
        orig_h = 480  # Original height before padding

        # Count total segments
        total_segments = 0
        for batch in segmentations:
            if batch is not None and batch.get('boxes') is not None:
                total_segments += len(batch['boxes'])

        # Draw dividing line between video and metrics area
        cv2.line(output_frame, (0, orig_h), (frame_w, orig_h), (128, 128, 128), 1)

        # Add summary info in the black area
        cv2.putText(output_frame, f"Segments detected: {total_segments}",
                    (10, orig_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Calculate metrics
        metrics = self.calculate_segmentation_metrics(segmentations)

        # Create overlay for masks
        mask_overlay = output_frame.copy()

        # For each segmentation batch
        segment_counter = 0
        for batch_idx, seg_batch in enumerate(segmentations):
            # Skip if the batch is None or empty
            if seg_batch is None:
                continue

            masks = seg_batch.get('masks')
            boxes = seg_batch.get('boxes')

            if masks is None or boxes is None:
                continue

            # For each segmentation in the batch
            for seg_idx in range(len(boxes)):
                try:
                    # Get box info
                    box = boxes[seg_idx]
                    x1, y1, x2, y2, conf, cls = box

                    # Only process segmentations above confidence threshold
                    if conf > 0.25:
                        # Get color for this class
                        color = self.class_colors[int(cls) % len(self.class_colors)]

                        # Draw mask if available
                        if seg_idx < len(masks):
                            mask = masks[seg_idx]
                            
                            # Convert mask to the right size if needed
                            if mask.shape != (frame_h, frame_w):
                                mask = cv2.resize(mask.astype(np.uint8), (frame_w, frame_h))
                            
                            # Create colored mask
                            colored_mask = np.zeros_like(mask_overlay)
                            colored_mask[mask > 0.5] = color
                            
                            # Apply mask to overlay
                            mask_overlay = cv2.addWeighted(mask_overlay, 1, colored_mask, alpha, 0)

                        # Draw bounding box outline
                        cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                        # Get class name from model
                        class_name = self.pt_model.names[int(cls)] if int(cls) < len(self.pt_model.names) else f"Class_{int(cls)}"

                        # Draw label with confidence
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        
                        # Draw label background
                        cv2.rectangle(output_frame, 
                                     (int(x1), int(y1) - label_size[1] - 10),
                                     (int(x1) + label_size[0], int(y1)), 
                                     color, -1)
                        
                        # Draw label text
                        cv2.putText(output_frame, label, (int(x1), int(y1) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        segment_counter += 1

                except (IndexError, ValueError):
                    continue

        # Blend the mask overlay with the original frame
        output_frame = cv2.addWeighted(output_frame, 1 - alpha, mask_overlay, alpha, 0)

        # Display metrics if requested in black area below original frame
        if show_metrics and metrics:
            metrics_x = 10
            metrics_y = orig_h + 50

            # Background for metrics panel
            panel_width = 300
            panel_height = 140
            cv2.rectangle(output_frame,
                         (metrics_x, metrics_y),
                         (metrics_x + panel_width, metrics_y + panel_height),
                         (0, 0, 0), -1)
            cv2.rectangle(output_frame,
                         (metrics_x, metrics_y),
                         (metrics_x + panel_width, metrics_y + panel_height),
                         (255, 255, 255), 1)

            # Title
            cv2.putText(output_frame, "Segmentation Metrics",
                       (metrics_x + 5, metrics_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Metrics
            y_offset = 35
            if 'avg_confidence' in metrics:
                cv2.putText(output_frame, f"Avg Conf: {metrics['avg_confidence']:.3f}",
                           (metrics_x + 5, metrics_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            if 'total_segments' in metrics:
                cv2.putText(output_frame, f"Total: {metrics['total_segments']}",
                           (metrics_x + 5, metrics_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            if 'high_conf_count' in metrics:
                cv2.putText(output_frame, f"High Conf (>0.8): {metrics['high_conf_count']}",
                           (metrics_x + 5, metrics_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            if 'avg_mask_area' in metrics:
                cv2.putText(output_frame, f"Avg Mask: {metrics['avg_mask_area']:.0f}px",
                           (metrics_x + 5, metrics_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            # Add mask transparency info
            cv2.putText(output_frame, f"Mask Alpha: {alpha:.1f}",
                       (metrics_x + 5, metrics_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output_frame

    def process_video(self, video_path, output_path=None, show_preview=True):
        """
        Process a video file to segment objects in each frame

        Args:
            video_path (str): Path to the input video file
            output_path (str, optional): Path for output video, or None to use auto naming
            show_preview (bool): Whether to display preview during processing

        Returns:
            tuple: (output_video_path, segmentations_data_path)
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepare output video writer (using square 640x640 output)
        if output_path is None:
            # Auto-generate output path with model name
            input_path = Path(video_path)
            model_suffix = self.short_model_name.replace('.pt', '')
            output_path = str(input_path.with_name(f"{input_path.stem}_{model_suffix}_segmentation{input_path.suffix}"))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 640))

        # Dictionary to store segmentations for all frames
        all_frames_segmentations = {}

        # Process frames
        frame_idx = 0
        start_time = time.time()
        coreml_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process this frame (resize if needed to 640x480)
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = frame[0:480, 0:640, :]

            # Process the frame and get segmentations and padded square frame
            segmentations, square_frame, used_coreml, current_fps = self.process_frame(frame)

            # Store segmentations with frame number as key
            all_frames_segmentations[frame_idx] = segmentations

            if used_coreml:
                coreml_frames += 1

            # Calculate average FPS
            elapsed_time = time.time() - start_time
            avg_fps = (frame_idx + 1) / elapsed_time if elapsed_time > 0 else 0

            # Calculate smooth FPS (moving average)
            smooth_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

            # Visualize and write to output video
            output_frame = self.visualize_segmentations(square_frame, segmentations)

            # Add frame and FPS info
            backend = "CoreML" if used_coreml else "PyTorch"
            cv2.putText(output_frame, f"Frame: {frame_idx}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(output_frame, f"FPS: {smooth_fps:.1f} ({backend})",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Write to output video
            out.write(output_frame)

            # Show preview if requested
            if show_preview:
                cv2.imshow("Segmentation", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Add screenshot capability
                    screenshot_path = str(Path(output_path).with_name(f"{Path(output_path).stem}_frame{frame_idx}.jpg"))
                    cv2.imwrite(screenshot_path, output_frame)
                    print(f"Screenshot saved: {screenshot_path}")

            # Print progress (less frequently)
            frame_idx += 1
            if frame_idx % 30 == 0:
                coreml_pct = (coreml_frames / frame_idx) * 100 if frame_idx > 0 else 0
                print(f"Processed {frame_idx}/{total_frames} frames - Avg FPS: {avg_fps:.1f} - CoreML: {coreml_pct:.1f}%")

            # After processing a few frames, if CoreML is failing, disable it to save time
            if frame_idx >= 10 and coreml_frames == 0 and self.coreml_model is not None:
                print("CoreML not working, switching to PyTorch only")
                self.coreml_model = None

        # Clean up
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        # Save segmentations to pickle file
        segmentations_path = str(Path(output_path).with_suffix('.pkl'))
        with open(segmentations_path, 'wb') as f:
            pickle.dump(all_frames_segmentations, f)

        print("\nVideo processing complete:")
        print(f"- Output video: {output_path}")
        print(f"- Segmentations data: {segmentations_path}")

        return output_path, segmentations_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Segmentation - Production Version")
    parser.add_argument("--model", type=str, default="yolo11m-seg",
                       help="Model to use (default: yolo11m-seg)")
    parser.add_argument("--cache-dir", type=str, default="./model_cache",
                       help="Directory to cache models")
    parser.add_argument("--video", type=str, required=False,
                       help="Path to the input video file")
    parser.add_argument("--output", type=str, default=None,
                       help="Path for output video (defaults to input_modelname_segmentation.mp4)")
    parser.add_argument("--force-export", action="store_true",
                       help="Force re-export of CoreML model even if it exists")
    parser.add_argument("--no-preview", action="store_true",
                       help="Disable preview window during processing")

    args = parser.parse_args()
    args.video = "/Users/prasoon/Research/Autonomous/autonomous-v2-mac/data_preprocessing/segmentation/sample5.mp4"
    args.cache_dir = "/Users/prasoon/Research/VAX/OrganicHAR/models/segmentation"
    # Initialize segmentation detector
    segmentation_detector = YoloSegmentation(
        model_name=args.model,
        cache_dir=args.cache_dir,
        force_export=args.force_export
    )

    # Process video
    output_path, segmentations_path = segmentation_detector.process_video(
        video_path=args.video,
        output_path=args.output,
        show_preview=not args.no_preview  # Preview enabled by default
    )

    print("Done!")


if __name__ == "__main__":
    main()
