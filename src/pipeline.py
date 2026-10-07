from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2

from .config import AppSettings
from .tracking.schemas import Detection, TrackingFrameOutput
from .tracking.track_manager import TrackManager
from .tracking.visualizer import draw_tracking
from .trajectory.predictor import TrajectoryPredictor

Perception = Callable[[Any, int, float], list[Detection]]


def run_video(
    video_path: Path,
    perception: Perception,
    settings: AppSettings,
    output_video: Path | None = None,
) -> list[TrackingFrameOutput]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_skip = max(1, settings.video.frame_skip)
    effective_fps = fps / frame_skip
    manager = TrackManager(settings.tracking)
    predictor = TrajectoryPredictor(
        effective_fps,
        settings.prediction.horizon_seconds,
        settings.prediction.sequence_length,
        settings.prediction.model_path,
        mode=settings.prediction.mode,
        scaler_path=settings.prediction.scaler_path,
    )
    writer = None
    outputs: list[TrackingFrameOutput] = []
    source_frame_id = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_id = source_frame_id
            source_frame_id += 1
            if frame_id % frame_skip != 0:
                continue
            if (
                settings.video.max_frames is not None
                and len(outputs) >= settings.video.max_frames
            ):
                break
            timestamp = frame_id / fps
            detections = perception(frame, frame_id, timestamp)
            tracked = manager.update(detections)
            predictions = [
                prediction
                for state in manager.active_states.values()
                if state.status == "active"
                if (prediction := predictor.predict(state)) is not None
            ]
            result = TrackingFrameOutput(
                frame_id=frame_id,
                timestamp=timestamp,
                tracked_objects=tracked,
                predictions=predictions,
            )
            outputs.append(result)
            if (
                settings.video.progress_interval
                and len(outputs) % settings.video.progress_interval == 0
            ):
                print(f"Processed {len(outputs)} frames (source frame {frame_id})")
            if output_video is not None:
                if writer is None:
                    output_video.parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                        effective_fps,
                        (width, height),
                    )
                writer.write(draw_tracking(frame, manager.active_states, predictions))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return outputs


def write_outputs(outputs: list[TrackingFrameOutput], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([output.model_dump() for output in outputs], indent=2),
        encoding="utf-8",
    )
