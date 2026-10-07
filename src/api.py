from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from .config import load_settings
from .perception.config import PerceptionConfig
from .perception.detector import YoloDetector
from .pipeline import run_video
from .tracking.schemas import TrackingFrameOutput

app = FastAPI(title="AI Safety Copilot - Tracking and Prediction")


class VideoProcessRequest(BaseModel):
    video_path: Path
    model_path: str = "yolo11s.pt"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "module": "person-2-tracking-prediction"}


@app.post("/tracking/frame", response_model=TrackingFrameOutput)
def tracking_frame(frame: TrackingFrameOutput) -> TrackingFrameOutput:
    """Contract endpoint for Person 3 integration and replay tooling."""
    return frame


@app.post("/tracking/process", response_model=list[TrackingFrameOutput])
def process_video(request: VideoProcessRequest) -> list[TrackingFrameOutput]:
    settings = load_settings()
    perception_config = PerceptionConfig(model_path=request.model_path)
    detector = YoloDetector(
        perception_config.model_path,
        perception_config.confidence,
        perception_config.iou,
        perception_config.image_size,
        perception_config.resolved_device(),
        perception_config.object_classes,
    )
    return run_video(
        request.video_path,
        lambda frame, frame_id, timestamp: detector.track_structured(
            frame, frame_id, timestamp
        ),
        settings,
    )
