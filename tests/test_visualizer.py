import numpy as np

from src.perception.visualizer import draw_detections


def test_draw_detections_preserves_frame_shape():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    objects = [
        {
            "class": "car",
            "bbox": [50, 60, 150, 160],
            "confidence": 0.9,
        }
    ]

    output = draw_detections(frame, objects)

    assert output.shape == frame.shape
    assert output.dtype == frame.dtype
    assert np.any(output != frame)
