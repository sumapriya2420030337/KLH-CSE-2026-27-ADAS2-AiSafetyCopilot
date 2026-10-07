from pathlib import Path

from src.perception.config import PerceptionConfig


def test_default_config():
    config = PerceptionConfig()

    assert config.input_dir == Path("data/videos")
    assert config.output_dir == Path("result/outputs")
    assert config.model_path == "yolo11s.pt"
    assert 0 < config.confidence < 1
    assert 0 < config.iou < 1
    assert config.image_size > 0


def test_explicit_cpu_device():
    config = PerceptionConfig(device="cpu")
    assert config.resolved_device() == "cpu"
