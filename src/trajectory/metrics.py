from __future__ import annotations

import torch


def displacement_errors(
    predicted: torch.Tensor, target: torch.Tensor
) -> tuple[float, float]:
    """Return ADE and FDE in the coordinate units of the inputs."""
    distances = torch.linalg.vector_norm(predicted - target, dim=-1)
    return float(distances.mean().item()), float(distances[:, -1].mean().item())
