import sys
import types
from typing import ClassVar

import numpy as np


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values

    def __iter__(self):
        return iter([FakeTensor(v) if isinstance(v, list) else v for v in self._values])

    def __float__(self):
        return float(self._values)


class FakeBox:
    def __init__(self):
        self.xyxy = FakeTensor([[10.0, 20.0, 100.0, 120.0]])
        self.conf = FakeTensor([0.9])
        self.cls = FakeTensor([2.0])
        self.id = FakeTensor([7.0])

    def __bool__(self):
        return True


class FakeResult:
    names: ClassVar[dict[int, str]] = {2: "car"}

    def __init__(self):
        self.boxes = FakeBox()


class FakeYOLO:
    def __init__(self, model_path):
        self.model_path = model_path
        self.track_kwargs = None

    def predict(self, **kwargs):
        return [FakeResult()]

    def track(self, **kwargs):
        self.track_kwargs = kwargs
        return [FakeResult()]


def test_detector_can_filter_and_format_detections(monkeypatch):
    fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    from src.perception.detector import YoloDetector

    detector = YoloDetector(
        model_path="fake.pt",
        confidence=0.25,
        iou=0.45,
        image_size=640,
        device="cpu",
        classes=("car",),
    )

    output = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(output) == 1
    assert output[0]["class"] == "car"
    assert output[0]["bbox"] == [10.0, 20.0, 100.0, 120.0]
    assert output[0]["confidence"] == 0.9


def test_detector_uses_persistent_ultralytics_bytetrack(monkeypatch):
    fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    from src.perception.detector import YoloDetector

    detector = YoloDetector("fake.pt", 0.25, 0.45, 416, "cpu", ("car",))
    output = detector.track_structured(np.zeros((100, 100, 3), dtype=np.uint8), 5, 0.5)

    assert output[0].tracker_id == 7
    assert detector.model.track_kwargs["persist"] is True
    assert detector.model.track_kwargs["conf"] == 0.1
    assert detector.model.track_kwargs["tracker"] == "bytetrack.yaml"
