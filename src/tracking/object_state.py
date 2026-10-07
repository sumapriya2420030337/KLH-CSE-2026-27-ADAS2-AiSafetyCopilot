from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .schemas import Detection, TrackedObject


@dataclass
class ObjectState:
    track_id: int
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    frame_id: int
    timestamp: float
    history_size: int = 30
    status: str = "active"
    missed_frames: int = 0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    last_seen_timestamp: float = 0.0
    confidence: float = 0.0
    observed_count: int = 0
    centers: deque[tuple[float, float]] = field(default_factory=deque)
    velocities: deque[tuple[float, float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.centers = deque(self.centers, maxlen=self.history_size)
        self.velocities = deque(self.velocities, maxlen=self.history_size)
        if self.last_seen_frame == 0 and self.first_seen_frame == 0:
            self.first_seen_frame = self.frame_id
        self.last_seen_frame = self.frame_id
        self.last_seen_timestamp = self.timestamp
        self.observed_count = max(1, self.observed_count)
        self._append_center(self.center)

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    @property
    def velocity(self) -> tuple[float, float]:
        return self.velocities[-1] if self.velocities else (0.0, 0.0)

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return (vx * vx + vy * vy) ** 0.5

    def _append_center(self, center: tuple[float, float]) -> None:
        previous = self.centers[-1] if self.centers else None
        previous_time = self.last_seen_timestamp
        self.centers.append(center)
        if previous is None:
            self.velocities.append((0.0, 0.0))
            return
        delta_time = max(self.timestamp - previous_time, 1e-6)
        self.velocities.append(
            (
                (center[0] - previous[0]) / delta_time,
                (center[1] - previous[1]) / delta_time,
            )
        )

    def update(self, detection: Detection) -> None:
        previous_timestamp = self.last_seen_timestamp
        self.bbox = detection.bbox
        self.frame_id = detection.frame_id
        self.timestamp = detection.timestamp
        self.confidence = detection.confidence
        self.observed_count += 1
        self.missed_frames = 0
        self.status = "active"
        previous = self.centers[-1] if self.centers else self.center
        self.centers.append(self.center)
        delta_time = max(detection.timestamp - previous_timestamp, 1e-6)
        self.velocities.append(
            (
                (self.center[0] - previous[0]) / delta_time,
                (self.center[1] - previous[1]) / delta_time,
            )
        )
        self.last_seen_frame = detection.frame_id
        self.last_seen_timestamp = detection.timestamp

    def predicted_bbox(self, timestamp: float) -> tuple[float, float, float, float]:
        """Project the last observed box to the current source timestamp."""
        elapsed = max(0.0, timestamp - self.last_seen_timestamp)
        offset_x = self.velocity[0] * elapsed
        offset_y = self.velocity[1] * elapsed
        return (
            self.bbox[0] + offset_x,
            self.bbox[1] + offset_y,
            self.bbox[2] + offset_x,
            self.bbox[3] + offset_y,
        )

    def mark_missed(self) -> None:
        self.missed_frames += 1
        self.status = "occluded"

    def as_output(self) -> TrackedObject:
        return TrackedObject(
            track_id=self.track_id,
            class_id=self.class_id,
            class_name=self.class_name,
            bbox=self.bbox,
            center=self.center,
            velocity=self.velocity,
            speed=self.speed,
            first_seen_frame=self.first_seen_frame,
            last_seen_frame=self.last_seen_frame,
            last_seen_timestamp=self.last_seen_timestamp,
            status=self.status,
            missed_frames=self.missed_frames,
            confidence=self.confidence,
        )
