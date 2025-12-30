from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig

from landscapes_classification import data_utilities


class LandscapesDataModule(pl.LightningDataModule):
    """
    Custom LightninDataModule implementation
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = data_utilities.init_dataset(
            self.cfg.data.train_dir,
            "train",
            self.cfg.train_params.img_size,
            self.cfg.train_params.norm_mean,
            self.cfg.train_params.norm_std,
        )
        self.val_dataset = data_utilities.init_dataset(
            self.cfg.data.val_dir,
            "val",
            self.cfg.train_params.img_size,
            self.cfg.train_params.norm_mean,
            self.cfg.train_params.norm_std,
        )
        self.test_dataset = data_utilities.init_dataset(
            self.cfg.data.test_dir,
            "test",
            self.cfg.train_params.img_size,
            self.cfg.train_params.norm_mean,
            self.cfg.train_params.norm_std,
        )

    def train_dataloader(self):
        return data_utilities.init_dataloader(
            self.train_dataset,
            self.cfg.train_params.train_batch_size,
            num_workers=self.cfg.train_params.num_workers,
        )

    def val_dataloader(self):
        return data_utilities.init_dataloader(
            self.val_dataset,
            self.cfg.train_params.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.train_params.num_workers,
        )

    def test_dataloader(self):
        return data_utilities.init_dataloader(
            self.test_dataset,
            self.cfg.train_params.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.train_params.num_workers,
        )
