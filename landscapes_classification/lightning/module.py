from typing import Any

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


class LandscapesModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = hydra.utils.instantiate(
            cfg.model[cfg.model.model_name], num_classes=cfg.train_params.num_classes
        )
        self.criterion = hydra.utils.instantiate(cfg.train_params.criterion)
        self.metric = hydra.utils.instantiate(
            cfg.train_params.metric, num_classes=cfg.train_params.num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Any):
        inputs, target = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, target)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        preds = outputs.argmax(dim=-1)
        accuracy = self.metric(preds, target)
        self.log(
            "train_accuracy",
            accuracy,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
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
        return hydra.utils.instantiate(self.cfg.train_params.optimizer, params=self.parameters())
