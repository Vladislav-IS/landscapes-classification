from typing import Any

import git
import hydra
import pytorch_lightning as pl
import torchmetrics
from models import BaselineNet, EfficientNet
from omegaconf import DictConfig
from torch import nn


class LandscapesModule(pl.LighningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        if cfg.model.model_name == "baseline":
            self.model = BaselineNet(3, cfg.model.hidden_size, cfg.model.num_calsses)
        else:
            self.model = EfficientNet(cfg.model.num_calsses)
        self.criterion = nn.CrossEntropyLoss()
        self.metric = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        try:
            repo = git.Repo(search_parent_directories=True)
            commit_id = repo.head.object.hexsha[:7]
            self.logger.experiment.set_tag("git_commit", commit_id)
        except Exception:
            self.logger.experiment.set_tag("git_commit", "unknown")

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
        return hydra.utils.instantiate(self.cfg.train_params.optimizer, params=self.parameters())
