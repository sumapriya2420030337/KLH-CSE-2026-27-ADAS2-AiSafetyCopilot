from collections.abc import Sequence
from typing import Any, cast


def draw_detections(frame: Any, objects: list[dict[str, object]]) -> Any:
    import cv2

    output = frame.copy()
    for item in objects:
        bbox_values = cast(Sequence[float], item["bbox"])
        bbox = [float(value) for value in bbox_values]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        confidence = float(cast(float, item["confidence"]))
        label = f"{str(item['class']).upper()} {confidence:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 120), 2)
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 220, 120),
            2,
        )
    return output
