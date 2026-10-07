from collections.abc import Sequence
from typing import Any, cast

from src.tracking.schemas import Detection


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        iou: float,
        image_size: int,
        device: str,
        classes: tuple[str, ...],
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required to run YOLO detection") from exc
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.device = device
        self.classes = set(classes)

    def detect(self, frame: Any) -> list[dict[str, object]]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        detections: list[dict[str, object]] = []
        for result in results:
            result = cast(Any, result)
            names = cast(dict[int, str], result.names)
            boxes = cast(Any, result.boxes)
            if boxes is None:
                continue
            for box, confidence, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
                class_id_value = int(cast(int | float, class_id))
                class_name = str(names[class_id_value])
                if class_name not in self.classes:
                    continue
                coordinates = [
                    max(0.0, float(value)) for value in cast(Any, box).tolist()
                ]
                if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
                    continue
                detections.append(
                    {
                        "class_id": class_id_value,
                        "class": class_name,
                        "bbox": coordinates,
                        "confidence": float(cast(float, confidence)),
                    }
                )
        return detections

    def detect_structured(
        self, frame: Any, frame_id: int, timestamp: float
    ) -> list[Detection]:
        detections: list[Detection] = []
        for item in self.detect(frame):
            bbox_values = cast(Sequence[float], item["bbox"])
            bbox = tuple(float(value) for value in bbox_values)
            detections.append(
                Detection(
                    class_id=int(cast(int | float, item.get("class_id", -1))),
                    class_name=str(item["class"]),
                    confidence=float(cast(float, item["confidence"])),
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )
        return detections

    def track_structured(
        self,
        frame: Any,
        frame_id: int,
        timestamp: float,
        tracker_config: str = "bytetrack.yaml",
    ) -> list[Detection]:
        """Run Ultralytics' maintained ByteTrack backend on one video frame."""
        results = self.model.track(
            source=frame,
            conf=0.1,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            tracker=tracker_config,
            persist=True,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            result = cast(Any, result)
            boxes = cast(Any, result.boxes)
            if boxes is None or boxes.id is None:
                continue
            names = cast(dict[int, str], result.names)
            for box, confidence, class_id, tracker_id in zip(
                boxes.xyxy,
                boxes.conf,
                boxes.cls,
                boxes.id,
            ):
                class_id_value = int(cast(int | float, class_id))
                class_name = str(names[class_id_value])
                if class_name not in self.classes:
                    continue
                coordinates = tuple(
                    max(0.0, float(value)) for value in cast(Any, box).tolist()
                )
                if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
                    continue
                bbox = cast(
                    tuple[float, float, float, float],
                    (coordinates[0], coordinates[1], coordinates[2], coordinates[3]),
                )
                detections.append(
                    Detection(
                        tracker_id=int(tracker_id),
                        class_id=class_id_value,
                        class_name=class_name,
                        confidence=float(cast(float, confidence)),
                        bbox=bbox,
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                )
        return detections
