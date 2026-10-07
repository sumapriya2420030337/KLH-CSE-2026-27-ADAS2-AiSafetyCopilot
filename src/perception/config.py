from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PerceptionConfig:
    input_dir: Path = Path("data/videos")
    output_dir: Path = Path("result/outputs")
    model_path: str = "yolo11s.pt"
    segmentation_model_path: str | None = None
    lane_model_path: str | None = None
    traffic_sign_model_path: str | None = None
    traffic_light_model_path: str | None = None
    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 640
    device: str = "auto"
    save_visualizations: bool = False
    object_classes: tuple[str, ...] = field(
        default_factory=lambda: (
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "truck",
        )
    )

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
