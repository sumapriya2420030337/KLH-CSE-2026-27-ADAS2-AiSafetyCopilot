from __future__ import annotations

import argparse
import json
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataset import TrajectoryDataset, group_by_track, load_observations
from .lstm_model import TrajectoryLSTM
from .metrics import displacement_errors
from .preprocessing import FeatureNormalizer


@dataclass
class TrainingConfig:
    sequence_length: int = 75
    prediction_steps: int = 75
    sampling_rate: float = 25.0
    history_seconds: float = 3.0
    prediction_seconds: float = 3.0
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 20
    seed: int = 42


def _arrays(observations: list) -> list[tuple[str, np.ndarray]]:
    trajectories = []
    for values in group_by_track(observations).values():
        trajectories.append(
            (
                values[0].source_id,
                np.asarray(
                    [
                        [item.center_x, item.center_y, item.velocity_x, item.velocity_y]
                        for item in values
                    ],
                    dtype=np.float32,
                ),
            )
        )
    return trajectories


def _split_tracks(
    trajectories: list[tuple[str, np.ndarray]], seed: int
) -> tuple[list[np.ndarray], ...]:
    by_source: dict[str, list[np.ndarray]] = {}
    for source, trajectory in trajectories:
        by_source.setdefault(source, []).append(trajectory)
    if len(by_source) < 3:
        raise ValueError(
            "at least three HighD recordings are required for source-level train/validation/test splits"
        )
    rng = np.random.default_rng(seed)
    sources = list(by_source)
    order = rng.permutation(len(sources))
    train_end = max(1, int(len(order) * 0.7))
    validation_end = min(len(order) - 1, max(train_end + 1, int(len(order) * 0.85)))
    return (
        [
            trajectory
            for index in order[:train_end]
            for trajectory in by_source[sources[index]]
        ],
        [
            trajectory
            for index in order[train_end:validation_end]
            for trajectory in by_source[sources[index]]
        ],
        [
            trajectory
            for index in order[validation_end:]
            for trajectory in by_source[sources[index]]
        ],
    )


def _evaluate(
    model: TrajectoryLSTM, dataset: Dataset, prediction_steps: int
) -> tuple[float, float, float]:
    if len(cast(Sized, dataset)) == 0:
        return float("nan"), float("nan"), float("nan")
    loader = DataLoader(dataset, batch_size=64)
    loss_function = nn.MSELoss()
    losses: list[float] = []
    ade_values: list[float] = []
    fde_values: list[float] = []
    model.eval()
    with torch.inference_mode():
        for inputs, targets in loader:
            predictions = model(inputs, prediction_steps)
            losses.append(float(loss_function(predictions, targets).item()))
            ade, fde = displacement_errors(predictions, targets)
            ade_values.append(ade)
            fde_values.append(fde)
    return (
        float(np.mean(losses)),
        float(np.mean(ade_values)),
        float(np.mean(fde_values)),
    )


def train_from_file(
    dataset_path: Path, output_dir: Path, config: TrainingConfig
) -> dict[str, float | int | str]:
    torch.manual_seed(config.seed)
    observations = load_observations(dataset_path)
    trajectories = _arrays(observations)
    train_trajectories, validation_trajectories, test_trajectories = _split_tracks(
        trajectories, config.seed
    )
    normalizer = FeatureNormalizer.fit(np.concatenate(train_trajectories, axis=0))
    train_set = TrajectoryDataset(
        train_trajectories, config.sequence_length, config.prediction_steps, normalizer
    )
    validation_set = TrajectoryDataset(
        validation_trajectories,
        config.sequence_length,
        config.prediction_steps,
        normalizer,
    )
    test_set = TrajectoryDataset(
        test_trajectories, config.sequence_length, config.prediction_steps, normalizer
    )
    if len(train_set) == 0:
        raise ValueError(
            "training trajectories do not contain a complete input/target window"
        )
    model = TrajectoryLSTM(
        input_size=4,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    best_validation = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "trajectory_lstm.pt"
    for _ in range(config.epochs):
        model.train()
        for inputs, targets in DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True
        ):
            optimizer.zero_grad()
            loss = loss_function(model(inputs, config.prediction_steps), targets)
            loss.backward()
            optimizer.step()
        validation_loss, _, _ = _evaluate(
            model, validation_set, config.prediction_steps
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": {
                        "input_size": 4,
                        "hidden_size": config.hidden_size,
                        "num_layers": config.num_layers,
                        "dropout": config.dropout,
                        "output_size": 2,
                    },
                },
                checkpoint_path,
            )
    np.savez(
        output_dir / "trajectory_lstm.scaler.npz",
        mean=normalizer.mean,
        scale=normalizer.scale,
    )
    test_loss, test_ade, test_fde = _evaluate(model, test_set, config.prediction_steps)
    summary: dict[str, float | int | str] = {
        "dataset": str(dataset_path),
        "train_sequences": len(train_set),
        "validation_sequences": len(validation_set),
        "test_sequences": len(test_set),
        "best_validation_loss": best_validation,
        "test_loss": test_loss,
        "test_ade_normalized": test_ade,
        "test_fde_normalized": test_fde,
        "checkpoint": str(checkpoint_path),
        "scaler": str(output_dir / "trajectory_lstm.scaler.npz"),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def train(
    trajectories: list[np.ndarray],
    sequence_length: int,
    prediction_steps: int,
    epochs: int,
    output_path: Path,
) -> None:
    """Compatibility entry point for callers already supplying trajectory arrays."""
    config = TrainingConfig(
        sequence_length=sequence_length,
        prediction_steps=prediction_steps,
        epochs=epochs,
    )
    dataset = TrajectoryDataset(trajectories, sequence_length, prediction_steps)
    if len(dataset) == 0:
        raise ValueError(
            "trajectory data does not contain a complete input/target window"
        )
    model = TrajectoryLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for _ in range(epochs):
        for inputs, targets in DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        ):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(inputs, prediction_steps), targets)
            loss.backward()
            optimizer.step()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {"input_size": 4, "hidden_size": 64, "num_layers": 2},
        },
        output_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the Person 2 LSTM trajectory predictor."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/trajectories.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/trajectory"))
    parser.add_argument("--history-seconds", type=float, default=3.0)
    parser.add_argument("--prediction-seconds", type=float, default=3.0)
    parser.add_argument("--sampling-rate", type=float, default=25.0)
    parser.add_argument("--history-length", type=int)
    parser.add_argument("--prediction-length", type=int)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if not args.dataset.exists():
        parser.error(
            f"trajectory dataset is missing: {args.dataset}; provide nuScenes/HighD-derived CSV or JSON"
        )
    summary = train_from_file(
        args.dataset,
        args.output_dir,
        TrainingConfig(
            sequence_length=args.history_length
            or round(args.history_seconds * args.sampling_rate),
            prediction_steps=args.prediction_length
            or round(args.prediction_seconds * args.sampling_rate),
            sampling_rate=args.sampling_rate,
            history_seconds=args.history_seconds,
            prediction_seconds=args.prediction_seconds,
            epochs=args.epochs,
        ),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
