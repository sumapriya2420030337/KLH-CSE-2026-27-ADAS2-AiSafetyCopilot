from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.tracking.object_state import ObjectState
from src.tracking.schemas import TrajectoryPrediction

from .lstm_model import TrajectoryLSTM
from .preprocessing import FeatureNormalizer


class TrajectoryPredictor:
    def __init__(
        self,
        fps: float,
        horizon_seconds: float,
        sequence_length: int,
        model_path: Path | None = None,
        device: str = "cpu",
        mode: str = "baseline",
        scaler_path: Path | None = None,
    ):
        self.fps = max(fps, 1.0)
        self.horizon_seconds = horizon_seconds
        self.sequence_length = sequence_length
        self.prediction_steps = max(1, round(self.fps * horizon_seconds))
        self.device = torch.device(device)
        self.mode = mode
        self.model: TrajectoryLSTM | None = None
        self.normalizer: FeatureNormalizer | None = None
        if mode not in ("baseline", "lstm"):
            raise ValueError("prediction mode must be baseline or lstm")
        if mode == "lstm" and (model_path is None or not model_path.exists()):
            raise FileNotFoundError(
                "LSTM prediction requested but no checkpoint was found; "
                "train a model and set prediction.model_path"
            )
        if mode == "lstm" and model_path is not None:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            self.model = TrajectoryLSTM(**checkpoint.get("model_config", {})).to(
                self.device
            )
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            scaler_file = scaler_path or model_path.with_suffix(".scaler.npz")
            if not scaler_file.exists():
                raise FileNotFoundError(
                    f"LSTM scaler not found next to checkpoint: {scaler_file}"
                )
            values = np.load(scaler_file)
            self.normalizer = FeatureNormalizer(values["mean"], values["scale"])

    def predict(self, state: ObjectState) -> TrajectoryPrediction | None:
        if state.status != "active" or len(state.centers) < self.sequence_length:
            return None
        raw_history = np.asarray(
            [
                [center[0], center[1], velocity[0], velocity[1]]
                for center, velocity in zip(state.centers, state.velocities)
            ],
            dtype=np.float32,
        )[-self.sequence_length :]
        if self.normalizer is not None:
            raw_history = self.normalizer.transform(raw_history)
        history = torch.tensor(
            raw_history,
            dtype=torch.float32,
            device=self.device,
        )
        if self.mode == "baseline":
            velocity = history[-1, 2:]
            steps = torch.arange(
                1, self.prediction_steps + 1, device=self.device, dtype=torch.float32
            ).unsqueeze(1)
            positions = history[-1, :2] + steps * velocity / self.fps
        else:
            if self.model is None:
                raise RuntimeError("LSTM predictor is not initialized")
            with torch.inference_mode():
                positions = self.model(history.unsqueeze(0), self.prediction_steps)[0]
            if self.normalizer is not None:
                positions = positions * torch.from_numpy(self.normalizer.scale[:2]).to(
                    self.device
                ) + torch.from_numpy(self.normalizer.mean[:2]).to(self.device)
        return TrajectoryPrediction(
            track_id=state.track_id,
            horizon_seconds=self.prediction_steps / self.fps,
            predicted_positions=[
                (float(position[0]), float(position[1])) for position in positions.cpu()
            ],
        )
