import cv2
import numpy as np
import pytest
import torch

from src.config import AppSettings
from src.pipeline import run_video
from src.tracking.schemas import Detection
from src.tracking.track_manager import TrackManager, TrackManagerConfig
from src.tracking.visualizer import draw_tracking
from src.trajectory.dataset import (
    TrajectoryDataset,
    TrajectoryObservation,
    group_by_track,
)
from src.trajectory.lstm_model import TrajectoryLSTM
from src.trajectory.metrics import displacement_errors
from src.trajectory.predictor import TrajectoryPredictor
from src.trajectory.prepare_highd import prepare_highd


def detection(frame: int, x: float = 10.0) -> Detection:
    return Detection(
        tracker_id=1,
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox=(x, 10, x + 10, 20),
        frame_id=frame,
        timestamp=frame / 10,
    )


def test_track_identity_and_occlusion_lifecycle():
    manager = TrackManager(TrackManagerConfig(max_missing_frames=2))
    first = manager.update([detection(0)])[0]
    second = manager.update([detection(1, 11)])[0]
    occluded = manager.update([])[0]
    assert first.track_id == second.track_id == occluded.track_id
    assert second.velocity[0] > 0
    assert occluded.status == "occluded"
    manager.update([])
    manager.update([])
    assert manager.get(first.track_id) is None


def test_association_is_class_consistent_and_terminates_tracks():
    manager = TrackManager(TrackManagerConfig(max_missing_frames=1))
    first = manager.update([detection(0)])[0]
    other_class = Detection(
        tracker_id=2,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(10, 10, 20, 20),
        frame_id=1,
        timestamp=0.1,
    )
    second = manager.update([other_class])
    assert {item.track_id for item in second} == {first.track_id, first.track_id + 1}
    assert manager.get(first.track_id).class_name == "car"
    manager.update([])
    manager.update([])
    assert manager.get(first.track_id) is None


def test_occluded_track_is_not_rendered_as_current_box():
    manager = TrackManager()
    manager.update([detection(0)])
    manager.update([])
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    rendered = draw_tracking(frame, manager.active_states, [])
    assert np.array_equal(rendered[10, 10], frame[10, 10])


def test_prediction_requires_active_sufficient_history():
    manager = TrackManager(TrackManagerConfig(history_size=10))
    for frame in range(4):
        manager.update([detection(frame, 10 + frame)])
    state = manager.get(1)
    predictor = TrajectoryPredictor(fps=10.0, horizon_seconds=3.0, sequence_length=3)
    assert state is not None
    assert predictor.predict(state) is not None
    manager.update([])
    assert predictor.predict(state) is None


def test_trajectory_grouping_preserves_ids_and_temporal_order():
    observations = [
        TrajectoryObservation(2, 2, "car", 2.0, 20, 2, 2, source_id="01"),
        TrajectoryObservation(1, 2, "car", 3.0, 30, 3, 3, source_id="01"),
        TrajectoryObservation(2, 2, "car", 1.0, 10, 1, 1, source_id="01"),
    ]
    grouped = group_by_track(observations)
    assert list(grouped) == [2, 1]
    assert [item.frame_id for item in grouped[2]] == [10, 20]
    assert all(item.track_id == 2 for item in grouped[2])


def test_ade_and_fde_are_position_errors():
    predicted = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    ade, fde = displacement_errors(predicted, target)
    assert ade == 0.5
    assert fde == 1.0


def test_lstm_mode_requires_checkpoint_and_scaler(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        TrajectoryPredictor(
            fps=10.0,
            horizon_seconds=3.0,
            sequence_length=3,
            mode="lstm",
            model_path=tmp_path / "missing.pt",
        )


def test_lstm_checkpoint_round_trip(tmp_path):
    checkpoint = tmp_path / "model.pt"
    scaler = tmp_path / "model.scaler.npz"
    model = TrajectoryLSTM(input_size=4, hidden_size=8, num_layers=1)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {"input_size": 4, "hidden_size": 8, "num_layers": 1},
        },
        checkpoint,
    )
    np.savez(
        scaler, mean=np.zeros(4, dtype=np.float32), scale=np.ones(4, dtype=np.float32)
    )
    manager = TrackManager()
    for frame in range(4):
        manager.update([detection(frame, 10 + frame)])
    predictor = TrajectoryPredictor(
        fps=10.0,
        horizon_seconds=0.3,
        sequence_length=3,
        model_path=checkpoint,
        mode="lstm",
    )
    result = predictor.predict(manager.get(1))
    assert result is not None
    assert len(result.predicted_positions) == 3


def test_prepare_highd_converts_official_track_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    header = "id,frame,x,y,width,height,xVelocity,yVelocity\n"
    (raw / "01_tracks.csv").write_text(
        header + "1,0,10,20,4,2,3,4\n1,25,11,21,4,2,3,4\n",
        encoding="utf-8",
    )
    (raw / "01_recordingMeta.csv").write_text("frameRate\n25\n", encoding="utf-8")
    (raw / "02_tracks.csv").write_text(header + "1,0,30,40,4,2,1,0\n", encoding="utf-8")
    (raw / "02_recordingMeta.csv").write_text("frameRate\n25\n", encoding="utf-8")
    output = tmp_path / "trajectories.csv"

    summary = prepare_highd(raw, output)
    rows = output.read_text(encoding="utf-8").splitlines()
    assert summary == {"recordings": 2, "observations": 3, "output": str(output)}
    assert rows[0].startswith("track_id,source_id")
    assert rows[1].split(",")[0] == "1000001"
    assert rows[1].split(",")[4] == "0.0"
    assert rows[1].split(",")[6:8] == ["12.0", "21.0"]
    assert rows[3].split(",")[0] == "2000001"


def test_dataset_and_lstm_shapes():
    values = np.arange(48, dtype=np.float32).reshape(12, 4)
    dataset = TrajectoryDataset([values], sequence_length=4, prediction_steps=3)
    inputs, targets = dataset[0]
    output = TrajectoryLSTM(hidden_size=8, num_layers=1)(inputs.unsqueeze(0), 3)
    assert tuple(inputs.shape) == (4, 4)
    assert tuple(targets.shape) == (3, 2)
    assert tuple(output.shape) == (1, 3, 2)


def test_frame_skip_preserves_source_time_and_limits_processed_frames(tmp_path):
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32)
    )
    for _ in range(12):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()

    settings = AppSettings()
    settings.video.frame_skip = 5
    settings.video.max_frames = 2
    settings.video.progress_interval = 0

    outputs = run_video(
        video_path,
        lambda frame, frame_id, timestamp: [detection(frame_id, 10.0 + frame_id)],
        settings,
    )

    assert [output.frame_id for output in outputs] == [0, 5]
    assert [output.timestamp for output in outputs] == [0.0, 0.5]
    assert outputs[-1].tracked_objects[0].velocity[0] == 10.0
