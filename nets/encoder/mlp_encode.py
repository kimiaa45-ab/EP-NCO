# RL/nets/encoder/mlp_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = None, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        out_dim = out_dim if out_dim is not None else hidden_dim

        layers = []
        d = in_dim
        for i in range(max(1, n_layers - 1)):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden_dim

        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

        # init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        return self.net(x)
