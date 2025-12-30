from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from landscapes_classification.data_utilities import dvc_pull
from landscapes_classification.lightning import data_module, user_callbacks
from landscapes_classification.lightning.module import LandscapesModule


def train_command(cfg: DictConfig) -> None:
    """
    Start training the model

    :param cfg: Hydra configuration
    :type cfg: DictConfig
    """
    repo_path = Path(__file__).parents[2]
    full_data_path = repo_path / cfg.data.train_dir
    if not full_data_path.is_dir() and not dvc_pull():
        return

    datamodule = data_module.LandscapesDataModule(cfg)
    module = LandscapesModule(cfg)

    logger = hydra.utils.instantiate(cfg.logging.train_logger)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(repo_path / cfg.callbacks.dir_path),
            filename=f"{cfg.callbacks.file_name}_{cfg.model.model_name}",
            monitor="val_loss",
            save_top_k=1,
            every_n_epochs=1,
        ),
        user_callbacks.GitCommitIdCallback(),
        user_callbacks.DrawPlotsCallback(repo_path / cfg.logging.plots.save_dir),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.train_params.num_epochs,
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        default_root_dir=cfg.train_params.hydra_output_dir,
    )

    trainer.fit(module, datamodule=datamodule)
    module = LandscapesModule.load_from_checkpoint(
        str(
            repo_path
            / cfg.callbacks.dir_path
            / f"{cfg.callbacks.file_name}_{cfg.model.model_name}.ckpt"
        ),
        weights_only=False,
    )
    torch.save(module.model.state_dict(), cfg.model.output_file)
