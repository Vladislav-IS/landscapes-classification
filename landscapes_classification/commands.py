from pathlib import Path

import fire
from hydra import compose, initialize_config_dir

from landscapes_classification.infer_command import infer_command
from landscapes_classification.train_command import train_command


class Commands:
    def train(self):
        with initialize_config_dir(
            version_base=None, config_dir=str(Path(__file__).parents[1] / "config")
        ):
            cfg = compose(config_name="config")
            train_command(cfg)

    def infer(self):
        with initialize_config_dir(
            version_base=None, config_dir=str(Path(__file__).parents[1] / "config")
        ):
            cfg = compose(config_name="config")
            infer_command(cfg)


if __name__ == "__main__":
    fire.Fire(Commands)
