from __future__ import annotations

from dataclasses import dataclass, field

from .object_state import ObjectState
from .schemas import Detection, TrackedObject
from .tracker import ByteTrack, ByteTrackConfig, iou


@dataclass
class TrackManagerConfig:
    history_size: int = 30
    max_missing_frames: int = 30
    tracker: ByteTrackConfig = field(default_factory=ByteTrackConfig)


class TrackManager:
    """Own persistent identities and lifecycle state for one video stream."""

    def __init__(self, config: TrackManagerConfig | None = None):
        self.config = config or TrackManagerConfig()
        self.tracker = ByteTrack(self.config.tracker)
        self.states: dict[int, ObjectState] = {}
        self._next_track_id = 1

    @property
    def active_states(self) -> dict[int, ObjectState]:
        return self.states

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        detections = self._deduplicate_detections(detections)
        detections_by_id = {
            detection.tracker_id: detection
            for detection in detections
            if detection.tracker_id is not None
        }
        for track_id, detection in detections_by_id.items():
            if track_id in self.states:
                state = self.states[track_id]
                if (
                    state.class_id == detection.class_id
                    and state.class_name == detection.class_name
                ):
                    state.update(detection)
                    continue
                state.mark_missed()
            self.states[track_id] = ObjectState(
                track_id=track_id,
                class_id=detection.class_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                frame_id=detection.frame_id,
                timestamp=detection.timestamp,
                history_size=self.config.history_size,
                confidence=detection.confidence,
            )
            self._next_track_id = max(self._next_track_id, track_id + 1)

        for track_id, state in list(self.states.items()):
            if track_id not in detections_by_id:
                state.mark_missed()
                if state.missed_frames > self.config.max_missing_frames:
                    state.status = "dead"
                    del self.states[track_id]
        return [state.as_output() for state in self.states.values()]

    def _deduplicate_detections(self, detections: list[Detection]) -> list[Detection]:
        """Prevent overlapping same-class detector boxes creating duplicate births."""
        retained: list[Detection] = []
        threshold = self.config.tracker.match_threshold
        for detection in sorted(
            detections, key=lambda item: item.confidence, reverse=True
        ):
            if any(
                existing.class_id == detection.class_id
                and existing.class_name == detection.class_name
                and iou(existing.bbox, detection.bbox) >= threshold
                for existing in retained
            ):
                continue
            retained.append(detection)
        return retained

    def get(self, track_id: int) -> ObjectState | None:
        return self.states.get(track_id)
