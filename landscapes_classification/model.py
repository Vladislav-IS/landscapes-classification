import torch.nn as nn


class BaselineNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5),
            nn.MaxPool2d(kernel_size=3),
            nn.Flatten(),
            nn.Linear(in_features=hidden_size, out_features=out_size),
        )

    def forward(self, x):
        return self.seq(x)
