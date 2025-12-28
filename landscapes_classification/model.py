from typing import Any

import pytorch_lightning as pl
import torchmetrics
from torch import nn, optim
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


class LandscapesModule(pl.LighningModule):
    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def train_step(self, batch: Any):
        inputs, target = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, target)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any):
        inputs, target = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, target)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        preds = outputs.argmax(dim=-1)
        accuracy = self.metric(preds, target)
        self.log(
            "val_accuracy",
            accuracy,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Any):
        inputs, target = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, target)
        self.log("test_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        preds = outputs.argmax(dim=-1)
        accuracy = self.metric(preds, target)
        self.log(
            "test_accuracy",
            accuracy,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)
