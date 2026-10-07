from typing import Any, cast


class CustomTrafficDetector:
    def __init__(
        self,
        model_path: str | None,
        confidence: float,
        iou: float,
        image_size: int,
        device: str,
    ):
        self.model = None
        if model_path:
            try:
                from ultralytics import YOLO

                self.model = YOLO(model_path)
            except ImportError as exc:
                raise RuntimeError(
                    "ultralytics is required for traffic recognition"
                ) from exc
        self.confidence, self.iou, self.image_size, self.device = (
            confidence,
            iou,
            image_size,
            device,
        )

    def detect(self, frame: Any) -> list[dict[str, object]]:
        if self.model is None:
            return []
        detections: list[dict[str, object]] = []
        for result in self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        ):
            result = cast(Any, result)
            boxes = cast(Any, result.boxes)
            if boxes is None:
                continue
            for box, confidence, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
                detections.append(
                    {
                        "class": str(result.names[int(class_id)]),
                        "bbox": [float(v) for v in cast(Any, box).tolist()],
                        "confidence": float(confidence),
                    }
                )
        return detections
