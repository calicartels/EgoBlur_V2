# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Optimized EgoBlur Gen2 video/image processing script.

Improvements over the original demo_ego_blur_gen2.py:
  1. Batch inference — process N frames per GPU forward pass
  2. GPU-only preprocessing — resize via F.interpolate, no CPU roundtrip
  3. GPU-accelerated blurring — torchvision gaussian_blur on GPU tensors
  4. Streaming output — ffmpeg pipe instead of in-memory frame list
  5. OpenCV VideoCapture — replaces moviepy for frame decoding
  6. Checkpointing — resume from last processed frame after interruption
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from gen2.script.constants import (
    FACE_THRESHOLDS_GEN2,
    LP_THRESHOLDS_GEN2,
    RESIZE_MAX_GEN2,
    RESIZE_MIN_GEN2,
)
from gen2.script.detectron2.export.torchscript_patch import patch_instances
from gen2.script.predictor import ClassID, EgoblurDetector, PATCH_INSTANCES_FIELDS
from gen2.script.utils import (
    get_device,
    setup_logger,
    validate_inputs,
)
from tqdm.auto import tqdm


logger = setup_logger()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimized EgoBlur Gen2 — batched, GPU-accelerated, streaming"
    )
    parser.add_argument(
        "--camera_name",
        required=False,
        type=str,
        default=None,
        choices=[
            "slam-front-left",
            "slam-front-right",
            "slam-side-left",
            "slam-side-right",
            "camera-rgb",
        ],
        help="Optional camera identifier for camera-specific default thresholds.",
    )
    parser.add_argument(
        "--face_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur Gen2 face model file path",
    )
    parser.add_argument(
        "--face_model_score_threshold",
        required=False,
        type=float,
        default=None,
        help="Face model score threshold to filter out low confidence detections",
    )
    parser.add_argument(
        "--lp_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur Gen2 license plate model file path",
    )
    parser.add_argument(
        "--lp_model_score_threshold",
        required=False,
        type=float,
        default=None,
        help="License plate model score threshold to filter out low confidence detections",
    )
    parser.add_argument(
        "--nms_iou_threshold",
        required=False,
        type=float,
        default=0.5,
        help="NMS iou threshold to filter out low confidence overlapping boxes",
    )
    parser.add_argument(
        "--scale_factor_detections",
        required=False,
        type=float,
        default=1,
        help="Scale detections by the given factor to allow blurring more area",
    )
    parser.add_argument(
        "--input_image_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path for the given image on which we want to make detections",
    )
    parser.add_argument(
        "--output_image_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path where we want to store the visualized image",
    )
    parser.add_argument(
        "--input_video_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path for the given video on which we want to make detections",
    )
    parser.add_argument(
        "--output_video_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path where we want to store the visualized video",
    )
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=8,
        help="Number of frames to batch together for GPU inference (default: 8)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        required=False,
        type=int,
        default=100,
        help="Write checkpoint every N frames (default: 100)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing checkpoint and start from scratch",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Threshold resolution (same as original)
# ---------------------------------------------------------------------------

def _get_threshold(
    camera_name: Optional[str],
    user_threshold: Optional[float],
    threshold_map: Optional[Dict[str, float]],
) -> float:
    if user_threshold is not None:
        return user_threshold
    if threshold_map is not None:
        if camera_name is not None:
            return threshold_map.get(camera_name, threshold_map["camera-rgb"])
        else:
            return threshold_map["camera-rgb"]
    raise ValueError(
        "Cannot retrieve the model score threshold. Please provide a user-specified "
        "threshold or a mapping from camera name to threshold."
    )


# ---------------------------------------------------------------------------
# GPU-accelerated blurring
# ---------------------------------------------------------------------------

def blur_regions_gpu(
    image_tensor: torch.Tensor,
    detections: List[List[float]],
    scale_factor: float,
) -> torch.Tensor:
    """
    Apply heavy Gaussian blur to detected regions using GPU tensors.

    Args:
        image_tensor: CHW uint8 tensor on GPU (BGR order).
        detections: List of [x1, y1, x2, y2] bounding boxes.
        scale_factor: Scale factor for detections.

    Returns:
        CHW uint8 tensor with detected regions blurred.
    """
    if not detections:
        return image_tensor

    C, H, W = image_tensor.shape

    # Compute blur kernel size — match original behavior (half image size)
    ksize = max(H // 2, 1) | 1  # must be odd
    if ksize % 2 == 0:
        ksize += 1
    sigma = ksize / 3.0

    # Create the blurred version of the full image on GPU
    img_float = image_tensor.unsqueeze(0).float()
    blurred = TF.gaussian_blur(img_float, kernel_size=[ksize, ksize], sigma=[sigma, sigma])
    blurred = blurred.squeeze(0).to(image_tensor.dtype)

    # Create mask on CPU (ellipse drawing needs OpenCV), then upload
    mask = np.zeros((H, W), dtype=np.uint8)

    for box in detections:
        if scale_factor != 1.0:
            box = _scale_box(box, W, H, scale_factor)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        cv2.ellipse(
            mask,
            (((x1 + x2) // 2, (y1 + y2) // 2), (w, h), 0),
            255,
            -1,
        )

    mask_tensor = torch.from_numpy(mask).to(image_tensor.device).bool()
    # Broadcast mask across channels: where mask is True, use blurred; else original
    result = torch.where(mask_tensor.unsqueeze(0), blurred, image_tensor)
    return result


def _scale_box(
    box: List[float], max_width: int, max_height: int, scale: float
) -> List[float]:
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    w = scale * w
    h = scale * h
    x1 = max(xc - w / 2, 0)
    y1 = max(yc - h / 2, 0)
    x2 = min(xc + w / 2, max_width)
    y2 = min(yc + h / 2, max_height)
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _checkpoint_path(output_video_path: str) -> str:
    return output_video_path + ".checkpoint.json"


def load_checkpoint(output_video_path: str) -> Optional[dict]:
    cp_path = _checkpoint_path(output_video_path)
    if os.path.exists(cp_path):
        with open(cp_path, "r") as f:
            return json.load(f)
    return None


def save_checkpoint(
    output_video_path: str, frames_written: int, total_frames: int, fps: float
):
    cp_path = _checkpoint_path(output_video_path)
    with open(cp_path, "w") as f:
        json.dump(
            {
                "frames_written": frames_written,
                "total_frames": total_frames,
                "fps": fps,
            },
            f,
        )


def delete_checkpoint(output_video_path: str):
    cp_path = _checkpoint_path(output_video_path)
    if os.path.exists(cp_path):
        os.remove(cp_path)


# ---------------------------------------------------------------------------
# Streaming video writer (cv2.VideoWriter — no RAM accumulation)
# ---------------------------------------------------------------------------

class StreamingVideoWriter:
    """Write BGR frames one-at-a-time via cv2.VideoWriter. Zero RAM accumulation."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        append: bool = False,
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps

        if append:
            # When resuming, write to a temp file — concatenate in close()
            self.temp_path = output_path + ".resume_part.mp4"
            target = self.temp_path
        else:
            self.temp_path = None
            target = output_path

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(target, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open: {target}")
        self.frames_written = 0

    def write_frame(self, bgr_frame: np.ndarray):
        if bgr_frame.dtype != np.uint8:
            bgr_frame = np.clip(bgr_frame, 0, 255).astype(np.uint8)
        self.writer.write(np.ascontiguousarray(bgr_frame))
        self.frames_written += 1

    def close(self):
        self.writer.release()

        if self.temp_path and os.path.exists(self.temp_path):
            # Concatenate original partial + resumed part via ffmpeg
            import subprocess
            concat_path = self.output_path + ".concat_list.txt"
            with open(concat_path, "w") as f:
                f.write(f"file '{os.path.abspath(self.output_path)}'\n")
                f.write(f"file '{os.path.abspath(self.temp_path)}'\n")
            merged = self.output_path + ".merged.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_path,
                    "-c", "copy",
                    merged,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            os.replace(merged, self.output_path)
            os.remove(self.temp_path)
            os.remove(concat_path)


# ---------------------------------------------------------------------------
# Image processing (single image — uses GPU blur)
# ---------------------------------------------------------------------------

def visualize_image(
    input_image_path: str,
    face_detector: Optional[EgoblurDetector],
    lp_detector: Optional[EgoblurDetector],
    output_image_path: str,
    scale_factor_detections: float,
):
    bgr_image = cv2.imread(input_image_path)
    if bgr_image is None:
        raise ValueError(f"Cannot read image: {input_image_path}")
    if len(bgr_image.shape) == 2:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    device = get_device()
    image_tensor = torch.from_numpy(
        np.transpose(bgr_image, (2, 0, 1))
    ).to(device)

    detections: List[List[float]] = []
    total_inference_time = 0.0

    with patch_instances(fields=PATCH_INSTANCES_FIELDS):
        if face_detector is not None:
            face_results = face_detector.run(image_tensor)
            total_inference_time += face_detector.last_inference_time
            if face_results and len(face_results) == 1:
                detections.extend(face_results[0])

        if lp_detector is not None:
            lp_results = lp_detector.run(image_tensor)
            total_inference_time += lp_detector.last_inference_time
            if lp_results and len(lp_results) == 1:
                detections.extend(lp_results[0])

    blur_start = time.time()
    result_tensor = blur_regions_gpu(image_tensor, detections, scale_factor_detections)
    result_bgr = result_tensor.cpu().numpy().transpose(1, 2, 0)
    blur_time = time.time() - blur_start

    logger.info("=" * 60)
    logger.info("SPEED REPORT (Image)")
    logger.info("=" * 60)
    logger.info(f"Inference time: {total_inference_time:.4f} seconds")
    logger.info(f"Blurring time:  {blur_time:.4f} seconds")
    logger.info(f"Total time:     {total_inference_time + blur_time:.4f} seconds")
    logger.info("=" * 60)

    cv2.imwrite(output_image_path, result_bgr)
    logger.info(f"Successfully wrote image to: {output_image_path}")


# ---------------------------------------------------------------------------
# Video processing — batched, streamed, checkpointed
# ---------------------------------------------------------------------------

def visualize_video(
    input_video_path: str,
    face_detector: Optional[EgoblurDetector],
    lp_detector: Optional[EgoblurDetector],
    output_video_path: str,
    scale_factor_detections: float,
    batch_size: int = 8,
    checkpoint_interval: int = 100,
    no_resume: bool = False,
) -> None:
    # --- Open video with OpenCV ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if not (isinstance(input_fps, (int, float)) and math.isfinite(input_fps) and input_fps > 0):
        cap.release()
        raise ValueError(
            f"Input video FPS is unavailable or invalid for {input_video_path}. "
            "Please provide a video file with a valid fixed FPS."
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = float(input_fps)

    # --- Checkpointing: resume if possible ---
    start_frame = 0
    append_mode = False
    if not no_resume:
        cp = load_checkpoint(output_video_path)
        if cp is not None and os.path.exists(output_video_path):
            start_frame = cp["frames_written"]
            if start_frame >= total_frames:
                logger.info("Checkpoint indicates all frames already processed. Skipping.")
                cap.release()
                delete_checkpoint(output_video_path)
                return
            logger.info(
                f"Resuming from frame {start_frame}/{total_frames} "
                f"({start_frame / total_frames * 100:.1f}% already done)"
            )
            append_mode = True

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # --- Open streaming video writer ---
    writer = StreamingVideoWriter(
        output_path=output_video_path,
        width=frame_width,
        height=frame_height,
        fps=output_fps,
        append=append_mode,
    )

    device = get_device()
    total_inference_time = 0.0
    total_blur_time = 0.0
    total_frame_time = 0.0
    frames_processed = 0
    global_frame_idx = start_frame

    # Graceful shutdown on Ctrl+C
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    remaining = total_frames - start_frame
    progress = tqdm(total=remaining, desc="Processing frames", unit="frame")

    try:
        with patch_instances(fields=PATCH_INSTANCES_FIELDS):
            while not interrupted:
                # --- Collect a batch of frames ---
                batch_bgr: List[np.ndarray] = []
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    batch_bgr.append(frame)

                if not batch_bgr:
                    break

                frame_start = time.time()
                B = len(batch_bgr)

                # --- Build batched GPU tensor (B, C, H, W) in BGR ---
                batch_np = np.stack(batch_bgr, axis=0)  # (B, H, W, C)
                batch_tensor = torch.from_numpy(
                    batch_np.transpose(0, 3, 1, 2)  # (B, C, H, W)
                ).to(device)

                # --- Run detectors with GPU preprocessing ---
                all_detections: List[List[List[float]]] = [[] for _ in range(B)]

                if face_detector is not None:
                    img_batch, orig_hw, model_hw = face_detector.pre_process_gpu(batch_tensor)

                    inf_start = time.time()
                    preds = face_detector.inference(img_batch)
                    inf_time = time.time() - inf_start
                    total_inference_time += inf_time

                    det_batch = face_detector.get_detections(
                        output_tensor=preds,
                        timestamp_s=0.0,
                        stream_id="",
                        rotation_angle=0.0,
                        model_input_hw_list=model_hw,
                        target_img_hw_list=orig_hw,
                    )
                    for i, det in enumerate(det_batch):
                        if det is not None:
                            all_detections[i].extend(det.face_bboxes.tolist())

                if lp_detector is not None:
                    img_batch, orig_hw, model_hw = lp_detector.pre_process_gpu(batch_tensor)

                    inf_start = time.time()
                    preds = lp_detector.inference(img_batch)
                    inf_time = time.time() - inf_start
                    total_inference_time += inf_time

                    det_batch = lp_detector.get_detections(
                        output_tensor=preds,
                        timestamp_s=0.0,
                        stream_id="",
                        rotation_angle=0.0,
                        model_input_hw_list=model_hw,
                        target_img_hw_list=orig_hw,
                    )
                    for i, det in enumerate(det_batch):
                        if det is not None:
                            all_detections[i].extend(det.lp_bboxes.tolist())

                # --- Blur + write each frame ---
                blur_start = time.time()
                for i in range(B):
                    frame_tensor = batch_tensor[i]  # (C, H, W) on GPU
                    result = blur_regions_gpu(
                        frame_tensor, all_detections[i], scale_factor_detections
                    )
                    result_bgr = result.cpu().numpy().transpose(1, 2, 0)
                    writer.write_frame(result_bgr)
                total_blur_time += time.time() - blur_start

                frame_time = time.time() - frame_start
                total_frame_time += frame_time
                frames_processed += B
                global_frame_idx += B
                progress.update(B)

                # --- Checkpoint ---
                if frames_processed % checkpoint_interval < batch_size:
                    save_checkpoint(
                        output_video_path, global_frame_idx, total_frames, output_fps
                    )

    except Exception:
        # Save checkpoint on any error so we can resume
        save_checkpoint(output_video_path, global_frame_idx, total_frames, output_fps)
        raise
    finally:
        progress.close()
        signal.signal(signal.SIGINT, original_handler)
        cap.release()
        writer.close()

    if interrupted:
        save_checkpoint(output_video_path, global_frame_idx, total_frames, output_fps)
        logger.info(
            f"Interrupted at frame {global_frame_idx}/{total_frames}. "
            f"Checkpoint saved — re-run the same command to resume."
        )
        return

    # --- Success: delete checkpoint ---
    delete_checkpoint(output_video_path)

    # --- Speed report ---
    if frames_processed > 0:
        avg_inf = total_inference_time / frames_processed
        avg_blur = total_blur_time / frames_processed
        avg_total = total_frame_time / frames_processed
        fps_achieved = frames_processed / total_frame_time if total_frame_time > 0 else 0

        logger.info("=" * 60)
        logger.info("SPEED REPORT")
        logger.info("=" * 60)
        logger.info(f"Total frames processed: {frames_processed}")
        logger.info(f"Batch size:             {batch_size}")
        logger.info(f"Inference speed:        {avg_inf:.4f} seconds/frame")
        logger.info(f"Blurring speed:         {avg_blur:.4f} seconds/frame")
        logger.info(f"Total speed:            {avg_total:.4f} seconds/frame")
        logger.info(f"Effective FPS:          {fps_achieved:.2f} frames/second")
        logger.info("-" * 60)
        logger.info(f"Total inference time:   {total_inference_time:.2f} seconds")
        logger.info(f"Total blur time:        {total_blur_time:.2f} seconds")
        logger.info(f"Total processing time:  {total_frame_time:.2f} seconds")
        logger.info("=" * 60)

    logger.info(f"Successfully output video to: {output_video_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = validate_inputs(parse_args())
    device = get_device()

    face_threshold = _get_threshold(
        args.camera_name, args.face_model_score_threshold, FACE_THRESHOLDS_GEN2
    )
    lp_threshold = _get_threshold(
        args.camera_name, args.lp_model_score_threshold, LP_THRESHOLDS_GEN2
    )

    face_detector: Optional[EgoblurDetector]
    if args.face_model_path is not None:
        face_detector = EgoblurDetector(
            model_path=args.face_model_path,
            device=device,
            detection_class=ClassID.FACE,
            score_threshold=face_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            resize_aug={
                "min_size_test": RESIZE_MIN_GEN2,
                "max_size_test": RESIZE_MAX_GEN2,
            },
        )
    else:
        face_detector = None

    lp_detector: Optional[EgoblurDetector]
    if args.lp_model_path is not None:
        lp_detector = EgoblurDetector(
            model_path=args.lp_model_path,
            device=device,
            detection_class=ClassID.LICENSE_PLATE,
            score_threshold=lp_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            resize_aug={
                "min_size_test": RESIZE_MIN_GEN2,
                "max_size_test": RESIZE_MAX_GEN2,
            },
        )
    else:
        lp_detector = None

    if args.input_image_path is not None:
        visualize_image(
            args.input_image_path,
            face_detector,
            lp_detector,
            args.output_image_path,
            args.scale_factor_detections,
        )

    if args.input_video_path is not None:
        visualize_video(
            args.input_video_path,
            face_detector,
            lp_detector,
            args.output_video_path,
            args.scale_factor_detections,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            no_resume=args.no_resume,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
