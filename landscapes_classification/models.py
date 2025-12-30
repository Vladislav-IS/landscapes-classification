from torch import nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


class BaselineNet(nn.Module):
    """
    Baseline neural network implementation
    """

    def __init__(self, hidden_size, adaptive_pool_size, num_classes):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(adaptive_pool_size),
            nn.Flatten(),
            nn.Linear(hidden_size * adaptive_pool_size**2, num_classes),
        )

    def forward(self, x):
        return self.seq(x)


class EfficientNetB3(nn.Module):
    """
    EfficientNet B3 implementation
    """

    def __init__(self, use_pretrained, num_classes):
        super().__init__()
        if use_pretrained:
            self.model = efficientnet_b3(EfficientNet_B3_Weights.IMAGENET1K_V1)
        else:
            self.model = efficientnet_b3()
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)
