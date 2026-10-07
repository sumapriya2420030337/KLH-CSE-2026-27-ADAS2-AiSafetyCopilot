from pathlib import Path
from typing import Any, cast


class SegmentationProcessor:
    """Optional custom segmentation; standard COCO weights do not mean drivable area."""

    def __init__(
        self, model_path: str | None, output_dir: Path, device: str, image_size: int
    ):
        self.output_dir = output_dir
        self.model = None
        self.device = device
        self.image_size = image_size
        if model_path:
            try:
                from ultralytics import YOLO

                self.model = YOLO(model_path)
            except ImportError as exc:
                raise RuntimeError("ultralytics is required for segmentation") from exc

    def process(
        self, frame: Any, video_stem: str, frame_id: int
    ) -> dict[str, object] | None:
        if self.model is None:
            return None
        results = self.model.predict(
            frame, device=self.device, imgsz=self.image_size, verbose=False
        )
        result_list = list(results)
        result = cast(Any, result_list[0]) if result_list else None
        if result is None or result.masks is None:
            return {"mask_available": False}
        mask_dir = self.output_dir / "masks" / video_stem
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"frame_{frame_id:06d}.npy"
        import numpy as np

        np.save(mask_path, result.masks.data[0].cpu().numpy())
        return {"mask_available": True, "mask_path": str(mask_path)}


class LaneDetector:
    """Lane model plug-in point. No lane coordinates are invented without a model."""

    def __init__(self, model_path: str | None):
        self.model_path = model_path

    def detect(self, frame: Any) -> list[dict[str, object]]:
        return []
