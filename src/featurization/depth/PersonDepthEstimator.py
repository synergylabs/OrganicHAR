import cv2
import numpy as np
import time
import pickle
from pathlib import Path
import argparse
"""
Developer Notes: I faced a very weird issue where the segementation model was failing to predict, when called from this class, but not when loaded independently. 
After few hours of debugging, I found that the issue was with the import order. If the import order is reversed(DepthEstimator first, then YoloSegmentation), it throws seg fault at YoloSegmentation:Line (self.coreml_model.predict(square_frame, verbose=False)).
I am not sure why this is happening.
"""
try:
    from .YoloSegmentation import YoloSegmentation
    from .MonocularDepthEstimation import DepthEstimator
except ImportError:
    from YoloSegmentation import YoloSegmentation
    from MonocularDepthEstimation import DepthEstimator


class PersonDepthEstimator:
    """
    A class to estimate depth specifically for persons in an image or video,
    by combining monocular depth estimation and person segmentation.
    """

    def __init__(self, depth_model_path="models/DepthAnythingV2SmallF16.mlpackage",
                 seg_model_name="yolov8n-seg", cache_dir="./model_cache"):
        """
        Initialize the PersonDepthEstimator.

        Args:
            depth_model_path (str): Path to the CoreML depth estimation model.
            seg_model_name (str): Name of the YOLO segmentation model.
            cache_dir (str): Directory to cache models.
        """
        print("Initializing Person Depth Estimator...")
        self.depth_estimator = DepthEstimator(model_path=depth_model_path)
        self.seg_estimator = YoloSegmentation(model_name=seg_model_name, cache_dir=cache_dir)

        # Get the class ID for 'person' from the segmentation model
        try:
            self.person_class_id = \
            [k for k, v in self.seg_estimator.pt_model.names.items() if v == 'person'][0]
        except IndexError:
            raise ValueError("The segmentation model does not have a 'person' class.")

        print(f"Person class ID is: {self.person_class_id}")
        print("Person Depth Estimator initialized.")

    def _align_depth_to_seg(self, raw_depth, original_frame_shape, target_shape=(640, 640)):
        """
        Align the raw depth map to the segmentation's coordinate space.
        """
        # This is complex. For now, we assume the depth model's output can be
        # directly resized to the segmentation output size.
        # A more robust solution would account for different padding/resizing strategies.
        aligned_depth = cv2.resize(raw_depth, target_shape, interpolation=cv2.INTER_NEAREST)
        return aligned_depth

    def get_person_mask(self, segmentations, target_shape=(640, 640)):
        """
        Extract a combined mask for all detected persons.
        """
        combined_mask = np.zeros(target_shape, dtype=np.uint8)
        for seg_batch in segmentations:
            if seg_batch is None or seg_batch.get('masks') is None or seg_batch.get('boxes') is None:
                continue

            masks = seg_batch['masks']
            boxes = seg_batch['boxes']

            for i, box in enumerate(boxes):
                if int(box[5]) == self.person_class_id:
                    mask = masks[i]
                    if mask.shape != target_shape:
                        mask = cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)
                    combined_mask = np.maximum(combined_mask, (mask > 0.5).astype(np.uint8))
        return combined_mask

    def process_frame(self, frame, output_raw=False):
        """
        Process a single frame to get depth for detected persons.

        Args:
            frame (numpy.ndarray): Input video frame (BGR format).

        Returns:
            tuple: (person_depth_raw, person_depth_viz, processing_time)
        """
        start_time = time.time()
        orig_h, orig_w = frame.shape[:2]

        # 1. Get segmentations (masks are relative to the padded 640x640 frame)
        segmentations, padded_frame, _, _ = self.seg_estimator.process_frame(frame)

        # 2. Get depth map for the whole frame (at the model's native output resolution)
        raw_depth, _, _ = self.depth_estimator.process_frame(frame)

        # 3. Create a combined mask for all detected persons in the padded space
        person_mask_padded = self.get_person_mask(segmentations, target_shape=(padded_frame.shape[0], padded_frame.shape[1]))

        # 4. Crop the segmentation mask from the padded space to the original frame's dimensions
        person_mask_unpadded = person_mask_padded[0:orig_h, 0:orig_w]

        # 5. Resize the raw depth map to match the original frame's dimensions
        depth_resized_to_frame = cv2.resize(raw_depth, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 6. Apply the unpadded person mask to the resized depth map
        person_depth_raw = depth_resized_to_frame * person_mask_unpadded

        # 7. Visualize the person-only depth
        person_depth_viz = self.depth_estimator.visualize_depth(person_depth_raw)

        processing_time = time.time() - start_time
        
        if output_raw:
            return person_depth_raw, person_depth_viz, processing_time, raw_depth, segmentations
        else:
            return person_depth_raw, person_depth_viz, processing_time

    def process_video(self, video_path, output_path=None, pickle_path=None, show_preview=True, output_fps=8):
        """
        Process a video to estimate depth for persons in each frame.

        Args:
            video_path (str): Path to the input video.
            output_path (str, optional): Path for the output video.
            pickle_path (str, optional): Path for the raw depth data pickle file.
            show_preview (bool): Whether to display a preview window.
            output_fps (int): Output FPS for the depth data extraction.
        Returns:
            str: Path to the output video file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, int(input_fps / output_fps))

        input_path = Path(video_path)
        if output_path is None:
            output_path = str(input_path.with_name(f"{input_path.stem}_person_depth.mp4"))
        if pickle_path is None:
            pickle_path = str(input_path.with_name(f"{input_path.stem}_person_depth_data.pickle"))

        # We process at 640x480, so the output should reflect that.
        proc_w, proc_h = 640, 480
        # out_w, out_h = proc_w * 2, proc_h
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (proc_w, proc_h))

        all_person_depths = []
        frame_idx = 0
        start_time = time.time()

        # print("Starting person depth video processing...")
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

            person_depth_raw, person_depth_viz, p_time = self.process_frame(frame)
            all_person_depths.append((frame_idx, person_depth_raw))

            elapsed_time = time.time() - start_time
            avg_fps = (frame_idx + 1) / elapsed_time if elapsed_time > 0 else 0

            # Add text overlays to visualization
            # cv2.putText(person_depth_viz, f"Frame: {frame_idx}/{total_frames}",
            #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(person_depth_viz, f"FPS: {avg_fps:.1f}",
            #             (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # combined_view = np.hstack((frame, person_depth_viz))
            # out.write(combined_view)
            out.write(person_depth_viz)

            if show_preview:
                cv2.imshow("Person Depth Estimation", person_depth_viz)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            # if frame_idx % 10 == 0:
            #     print(f"Processed {frame_idx}/{total_frames} frames... Avg FPS: {avg_fps:.1f}")

        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        # print(f"\nSaving person depth data to pickle file: {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_person_depths, f)

        # print("\nProcessing complete.")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Person Depth Estimation Video Processor")
    parser.add_argument("--depth-model", type=str, default="models/DepthAnythingV2SmallF16.mlpackage",
                        help="Path to CoreML depth model.")
    parser.add_argument("--seg-model", type=str, default="yolo11m-seg",
                        help="Name of the YOLO segmentation model.")
    parser.add_argument("--video", type=str, help="Path to the input video file.")
    parser.add_argument("--output", type=str, default=None, help="Path for output video.")
    parser.add_argument("--pickle", type=str, default=None, help="Path for raw depth data pickle.")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview window.")

    args = parser.parse_args()
    args.video = "/Users/prasoon/Research/Autonomous/autonomous-v2-mac/data_preprocessing/segmentation/sample5.mp4"
    args.depth_model = "/Users/prasoon/Research/VAX/OrganicHAR/models/monodepth/DepthAnythingV2SmallF16.mlpackage"
    args.seg_model = "yolo11m-seg"
    args.cache_dir = "/Users/prasoon/Research/VAX/OrganicHAR/models/segmentation"

    estimator = PersonDepthEstimator(
        depth_model_path=args.depth_model,
        seg_model_name=args.seg_model,
        cache_dir=args.cache_dir
    )

    estimator.process_video(
        video_path=args.video,
        output_path=args.output,
        pickle_path=args.pickle,
        show_preview=not args.no_preview
    )


if __name__ == "__main__":
    main() 