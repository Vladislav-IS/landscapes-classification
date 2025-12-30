import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from landscapes_classification.data_module import LandscapesDataModule
from landscapes_classification.module import LandscapesModule
from landscapes_classification.user_callbacks import GitCommitIdCallback


def infer_command(cfg: DictConfig) -> None:
    datamodule = LandscapesDataModule(cfg)
    module = LandscapesModule(cfg)
    module.model.load_state_dict(torch.load(cfg.model.output_file, weights_only=True))
    logger = hydra.utils.instantiate(cfg.logging.test_logger)
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[GitCommitIdCallback()],
        default_root_dir=cfg.train_params.hydra_output_dir,
    )
    trainer.test(module, datamodule=datamodule)
