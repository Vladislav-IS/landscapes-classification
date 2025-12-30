from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from landscapes_classification.data_utilities import dvc_pull
from landscapes_classification.lightning import data_module, user_callbacks
from landscapes_classification.lightning.module import LandscapesModule


def infer_command(cfg: DictConfig) -> None:
    """
    Start inferencing the model on the test data

    :param cfg: Hydra configuration
    :type cfg: DictConfig
    """
    repo_path = Path(__file__).parents[2]
    full_data_path = repo_path / cfg.data.train_dir
    if not full_data_path.is_dir() and not dvc_pull():
        return
    datamodule = data_module.LandscapesDataModule(cfg)
    module = LandscapesModule(cfg)
    module.model.load_state_dict(torch.load(cfg.model.output_file, weights_only=True))
    logger = hydra.utils.instantiate(cfg.logging.test_logger)
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[user_callbacks.GitCommitIdCallback()],
        default_root_dir=cfg.train_params.hydra_output_dir,
    )
    trainer.test(module, datamodule=datamodule)
