from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Detection(BaseModel):
    class_id: int = -1
    tracker_id: int | None = None
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]
    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0.0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if value[2] <= value[0] or value[3] <= value[1]:
            raise ValueError("bbox must be an increasing x1, y1, x2, y2 rectangle")
        return value

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]


class TrackedObject(BaseModel):
    track_id: int
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    velocity: tuple[float, float]
    speed: float = Field(ge=0.0)
    speed_unit: str = "pixels/second"
    first_seen_frame: int
    last_seen_frame: int
    last_seen_timestamp: float
    status: str
    missed_frames: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)


class TrajectoryPrediction(BaseModel):
    track_id: int
    horizon_seconds: float = Field(gt=0.0)
    predicted_positions: list[tuple[float, float]]


class VideoInfo(BaseModel):
    fps: float = Field(ge=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(ge=0)


class TrackingFrameOutput(BaseModel):
    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0.0)
    tracked_objects: list[TrackedObject]
    predictions: list[TrajectoryPrediction]
