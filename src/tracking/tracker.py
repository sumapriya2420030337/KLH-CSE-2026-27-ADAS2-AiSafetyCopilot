from __future__ import annotations

from dataclasses import dataclass


def iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


@dataclass
class ByteTrackConfig:
    high_confidence: float = 0.5
    low_confidence: float = 0.1
    match_threshold: float = 0.3
    track_buffer: int = 30


class ByteTrack:
    """Configuration descriptor for Ultralytics' maintained ByteTrack backend."""

    def __init__(self, config: ByteTrackConfig | None = None):
        self.config = config or ByteTrackConfig()

    @staticmethod
    def iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        return iou(first, second)
