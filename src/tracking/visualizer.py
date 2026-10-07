from __future__ import annotations

from itertools import pairwise
from typing import Any

import cv2

from .object_state import ObjectState
from .schemas import TrajectoryPrediction


def draw_tracking(
    frame: Any, states: dict[int, ObjectState], predictions: list[TrajectoryPrediction]
) -> Any:
    output = frame.copy()
    colors = [(40, 220, 120), (255, 170, 40), (180, 80, 230), (40, 190, 240)]
    for index, state in enumerate(states.values()):
        color = colors[index % len(colors)]
        if state.status != "active":
            center = tuple(int(value) for value in state.center)
            cv2.putText(
                output,
                f"{state.class_name} #{state.track_id} OCCLUDED ({state.missed_frames})",
                (center[0] + 8, max(18, center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1,
            )
            continue
        x1, y1, x2, y2 = [int(value) for value in state.bbox]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{state.class_name} #{state.track_id} {state.status}"
        cv2.putText(
            output,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        points = [(int(x), int(y)) for x, y in state.centers]
        for first, second in pairwise(points):
            cv2.line(output, first, second, color, 2)
    for prediction in predictions:
        points = [(int(x), int(y)) for x, y in prediction.predicted_positions]
        for first, second in pairwise(points):
            cv2.line(output, first, second, (0, 230, 255), 2, cv2.LINE_AA)
    return output
