from pathlib import Path

import git
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl


class GitCommitIdCallback(pl.callbacks.Callback):
    def __init__(self):
        try:
            repo = git.Repo(search_parent_directories=True)
            commit_id = repo.head.object.hexsha[:7]
            self.git_commit = commit_id
        except Exception:
            print("Warning: Error occured while extracting git commit. Set commit id to unknown")
            self.git_commit = "unknown"

    def _log_commit_id(self, trainer: pl.Trainer):
        try:
            trainer.logger.log_hyperparams({"git_commit": self.git_commit})
        except Exception:
            print("Warning: Error occured while trying to log git commit")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_commit_id(trainer)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_commit_id(trainer)


class DrawPlotsCallback(pl.callbacks.Callback):
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        fig, ax = plt.subplots(2, 2)
        fig.suptitle("Losses and metrics")
        if isinstance(trainer.logger, pl.loggers.MLFlowLogger):
            run_id = trainer.logger.run_id
            client = trainer.logger.experiment
            train_loss_history = client.get_metric_history(run_id, key="train_loss")
            train_acc_history = client.get_metric_history(run_id, key="train_accuracy")
            val_loss_history = client.get_metric_history(run_id, key="val_loss")
            val_acc_history = client.get_metric_history(run_id, key="val_accuracy")
            if (
                not train_loss_history
                or not train_acc_history
                or not val_loss_history
                or not val_acc_history
            ):
                print(f"Warning: Didn't find all metrics in MLFlow-runner {run_id}")
                return
            ax[0][0].plot([m.value for m in train_loss_history])
            ax[0][0].set_title("Train Loss")
            ax[0][1].plot([m.value for m in val_loss_history])
            ax[0][1].set_title("Validation Loss")
            ax[1][0].plot([m.value for m in train_acc_history])
            ax[1][0].set_title("Train Accuracy")
            ax[1][1].plot([m.value for m in val_acc_history])
            ax[1][1].set_title("Validation Accuracy")
        elif isinstance(trainer.logger, pl.loggers.CSVLogger):
            try:
                csv_file = Path(trainer.logger.log_dir) / "metrics.csv"
                metrics = pd.read_csv(csv_file)
            except Exception:
                print("Warning: Cannot find CSV-file with losses and metrics")
                return
            epochs = metrics["epoch"]
            train_loss_history = metrics["train_loss"].to_list()
            train_acc_history = metrics["train_accuracy"].to_list()
            val_loss_history = metrics["val_loss"].to_list()
            val_acc_history = metrics["val_accuracy"].to_list()
            if (
                not train_loss_history
                or not train_acc_history
                or not val_loss_history
                or not val_acc_history
            ):
                print("Warning: Didn't find all metrics in CSV-file")
                return
            ax[0][0].plot(epochs, train_loss_history)
            ax[0][0].set_title("Train Loss")
            ax[0][1].plot(epochs, val_loss_history)
            ax[0][1].set_title("Validation Loss")
            ax[1][0].plot(epochs, train_acc_history)
            ax[1][0].set_title("Train Accuracy")
            ax[1][1].plot(epochs, val_acc_history)
            ax[1][1].set_title("Validation Accuracy")
        else:
            print("Warning: valid loggers not found")
            return
        for row in range(2):
            for col in range(2):
                ax[row][col].set_xlabel("Epoch")
                ax[row][col].set_ylabel("Value")
        plt.tight_layout()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(self.save_dir / "metrics_train.png"))
        plt.close(fig)
        print(f"Info: Plots saved in directory: {self.save_dir}")
