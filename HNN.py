import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class HNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(2, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 1),
        )

    def forward(self, x):
        H = self.linear_block(x)
        return H
