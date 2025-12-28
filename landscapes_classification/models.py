from torch import nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


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


class EfficientNet(nn.Module):
    def __init__(self, out_size):
        self.model = efficientnet_b3(EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, num_classes=out_size
        )

    def forward(self, x):
        return self.model(x)
