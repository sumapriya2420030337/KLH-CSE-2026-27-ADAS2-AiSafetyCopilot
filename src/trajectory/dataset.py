from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import FeatureNormalizer


@dataclass(frozen=True)
class TrajectoryObservation:
    track_id: int
    class_id: int
    class_name: str
    timestamp: float
    frame_id: int
    center_x: float
    center_y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed: float = 0.0
    source_id: str = ""


def load_observations(path: Path) -> list[TrajectoryObservation]:
    """Load normalized observations from CSV or a flat JSON list."""
    if not path.exists():
        raise FileNotFoundError(f"trajectory dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("trajectory dataset must be CSV or JSON")
    required = {"track_id", "timestamp", "frame_id", "center_x", "center_y"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"trajectory dataset requires columns: {sorted(required)}")
    return [
        TrajectoryObservation(
            track_id=int(row["track_id"]),
            source_id=str(row.get("source_id", "")),
            class_id=int(row.get("class_id", -1)),
            class_name=str(row.get("class_name", "unknown")),
            timestamp=float(row["timestamp"]),
            frame_id=int(row["frame_id"]),
            center_x=float(row["center_x"]),
            center_y=float(row["center_y"]),
            velocity_x=float(row.get("velocity_x", 0.0)),
            velocity_y=float(row.get("velocity_y", 0.0)),
            speed=float(row.get("speed", 0.0)),
        )
        for row in rows
    ]


def group_by_track(
    observations: Iterable[TrajectoryObservation],
) -> dict[int, list[TrajectoryObservation]]:
    grouped: dict[int, list[TrajectoryObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.track_id, []).append(observation)
    for values in grouped.values():
        values.sort(key=lambda item: (item.timestamp, item.frame_id))
    return grouped


class TrajectoryDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Turn per-object [x, y, vx, vy] histories into leakage-free windows."""

    def __init__(
        self,
        trajectories: Iterable[np.ndarray],
        sequence_length: int,
        prediction_steps: int,
        normalizer: FeatureNormalizer | None = None,
    ):
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.normalizer = normalizer
        for trajectory in trajectories:
            values = np.asarray(trajectory, dtype=np.float32)
            if values.ndim != 2 or values.shape[1] < 4:
                raise ValueError("each trajectory must have shape [timesteps, 4]")
            if normalizer is not None:
                values = normalizer.transform(values)
            for end in range(sequence_length, len(values) - prediction_steps + 1):
                self.samples.append(
                    (
                        values[end - sequence_length : end, :4],
                        values[end : end + prediction_steps, :2],
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = self.samples[index]
        return torch.from_numpy(inputs), torch.from_numpy(targets)
