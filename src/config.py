from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from .tracking.track_manager import TrackManagerConfig
from .tracking.tracker import ByteTrackConfig


@dataclass
class PredictionSettings:
    mode: str = "baseline"
    sequence_length: int = 20
    horizon_seconds: float = 3.0
    hidden_size: int = 64
    num_layers: int = 2
    model_path: Path | None = None
    scaler_path: Path | None = None


@dataclass
class VideoSettings:
    frame_skip: int = 1
    imgsz: int = 416
    max_frames: int | None = None
    progress_interval: int = 10


@dataclass
class AppSettings:
    output_path: Path = Path("result/outputs")
    video: VideoSettings = field(default_factory=VideoSettings)
    tracking: TrackManagerConfig = field(default_factory=TrackManagerConfig)
    prediction: PredictionSettings = field(default_factory=PredictionSettings)


def load_settings(path: Path | None = None) -> AppSettings:
    if path is None or not path.exists():
        return AppSettings()
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    video = values.get("video", {})
    tracking = values.get("tracking", {})
    prediction = values.get("prediction", {})
    return AppSettings(
        output_path=Path(values.get("video", {}).get("output_path", "result/outputs")),
        video=VideoSettings(
            frame_skip=max(1, int(video.get("frame_skip", 1))),
            imgsz=max(32, int(video.get("imgsz", 416))),
            max_frames=(
                max(1, int(video["max_frames"]))
                if video.get("max_frames") is not None
                else None
            ),
            progress_interval=max(1, int(video.get("progress_interval", 10))),
        ),
        tracking=TrackManagerConfig(
            history_size=int(tracking.get("history_size", 30)),
            max_missing_frames=int(tracking.get("max_missing_frames", 30)),
            tracker=ByteTrackConfig(
                high_confidence=float(tracking.get("high_confidence", 0.5)),
                low_confidence=float(tracking.get("low_confidence", 0.1)),
                match_threshold=float(tracking.get("match_threshold", 0.3)),
                track_buffer=int(tracking.get("track_buffer", 30)),
            ),
        ),
        prediction=PredictionSettings(
            mode=str(prediction.get("mode", "baseline")),
            sequence_length=int(prediction.get("sequence_length", 20)),
            horizon_seconds=float(prediction.get("horizon_seconds", 3.0)),
            hidden_size=int(prediction.get("hidden_size", 64)),
            num_layers=int(prediction.get("num_layers", 2)),
            model_path=(
                Path(prediction["model_path"]) if prediction.get("model_path") else None
            ),
            scaler_path=(
                Path(prediction["scaler_path"])
                if prediction.get("scaler_path")
                else None
            ),
        ),
    )
