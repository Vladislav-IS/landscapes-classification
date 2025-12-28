import hydra
import pytorch_lightning as pl
from datamodule import LandscapesDataModule
from module import LandscapesModule
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    datamodule = LandscapesDataModule(cfg)
    module = LandscapesModule(cfg)

    loggers = [
        MLFlowLogger(
            tracking_uri=cfg.logging.mlflow.tracking_uri,
            experiment_name=cfg.logging.mlflow.experiment_name,
            save_dir=cfg.logging.mlflow.save_dir,
            run_name=cfg.logging.mlflow.run_name,
        ),
        TensorBoardLogger(
            save_dir=cfg.logging.tensorboard.save_dir, name=cfg.logging.tensorboard.name
        ),
    ]

    callbacks = ModelCheckpoint(
        dirpath=cfg.callbacks.dir_path,
        filename=cfg.callbacks.file_name,
        monitor="val_loss",
        save_top_k=1,
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train_params.num_epochs,
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
