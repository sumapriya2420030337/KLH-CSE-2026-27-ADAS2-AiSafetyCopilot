import json
from pathlib import Path

import cv2

from .config import PerceptionConfig
from .detector import YoloDetector
from .scene import LaneDetector, SegmentationProcessor
from .traffic import CustomTrafficDetector
from .visualizer import draw_detections


class VideoProcessor:
    def __init__(self, config: PerceptionConfig):
        self.config = config
        device = config.resolved_device()
        self.detector = YoloDetector(
            config.model_path,
            config.confidence,
            config.iou,
            config.image_size,
            device,
            config.object_classes,
        )
        self.segmentation = SegmentationProcessor(
            config.segmentation_model_path, config.output_dir, device, config.image_size
        )
        self.lanes = LaneDetector(config.lane_model_path)
        self.signs = CustomTrafficDetector(
            config.traffic_sign_model_path,
            config.confidence,
            config.iou,
            config.image_size,
            device,
        )
        self.lights = CustomTrafficDetector(
            config.traffic_light_model_path,
            config.confidence,
            config.iou,
            config.image_size,
            device,
        )

    def process_video(self, video_path: Path) -> Path | None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"ERROR: unable to open {video_path}")
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0 else 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: list[dict[str, object]] = []
        writer = None
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame_id = len(frames)
                objects = self.detector.detect(frame)
                item: dict[str, object] = {
                    "video_name": video_path.name,
                    "frame_id": frame_id,
                    "timestamp": frame_id / fps if fps else 0.0,
                    "objects": objects,
                    "traffic_signs": self.signs.detect(frame),
                    "traffic_lights": self.lights.detect(frame),
                    "lanes": self.lanes.detect(frame),
                    "drivable_area": self.segmentation.process(
                        frame, video_path.stem, frame_id
                    ),
                }
                frames.append(item)
                if self.config.save_visualizations:
                    if writer is None:
                        output_path = (
                            self.config.output_dir / f"{video_path.stem}_processed.mp4"
                        )
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                            fps or 30.0,
                            (width, height),
                        )
                    writer.write(draw_detections(frame, objects))
        finally:
            capture.release()
            if writer is not None:
                writer.release()
        output_path = self.config.output_dir / f"{video_path.stem}_detections.json"
        output_path.write_text(
            json.dumps(
                {
                    "video_name": video_path.name,
                    "fps": fps,
                    "frame_count": frame_count,
                    "frames": frames,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return output_path
