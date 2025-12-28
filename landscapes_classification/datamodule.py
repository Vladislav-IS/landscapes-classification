from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig

from data import init_dataloader, init_dataset


class LandscapesDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = init_dataset(
            self.cfg.data.train_dir, "train", self.cfg.train_params.img_size
        )
        self.val_dataset = init_dataset(
            self.cfg.data.val_dir, "val", self.cfg.train_params.img_size
        )
        self.test_dataset = init_dataset(
            self.cfg.data.test_dir, "test", self.cfg.train_params.img_size
        )

    def train_dataloader(self):
        return init_dataloader(
            self.train_dataset,
            self.cfg.train_params.batch_size,
            num_workers=self.cfg.train_params.num_workers,
        )

    def val_dataloader(self):
        return init_dataloader(
            self.val_dataset,
            self.cfg.train_params.batch_size,
            shuffle=False,
            num_workers=self.cfg.train_params.num_workers,
        )

    def test_dataloader(self):
        return init_dataloader(
            self.test_dataset,
            self.cfg.train_params.batch_size,
            shuffle=False,
            num_workers=self.cfg.train_params.num_workers,
        )
