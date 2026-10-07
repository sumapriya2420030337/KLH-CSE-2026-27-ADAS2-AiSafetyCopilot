from __future__ import annotations

import torch
from torch import nn


class TrajectoryLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        output_size: int = 2,
    ):
        super().__init__()
        if input_size not in (2, 4):
            raise ValueError("input_size must be 2 ([x,y]) or 4 ([x,y,vx,vy])")
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, history: torch.Tensor, future_steps: int) -> torch.Tensor:
        _, hidden = self.encoder(history)
        decoder_input = history[:, -1:, :]
        last_position = history[:, -1:, :2]
        predictions: list[torch.Tensor] = []
        for _ in range(future_steps):
            decoded, hidden = self.decoder(decoder_input, hidden)
            position = self.output(decoded[:, -1])
            predictions.append(position)
            velocity = position - last_position[:, 0]
            decoder_values = torch.cat((position, velocity), dim=-1)
            decoder_input = decoder_values[:, : self.input_size].unsqueeze(1)
            last_position = position.unsqueeze(1)
        return torch.stack(predictions, dim=1)
