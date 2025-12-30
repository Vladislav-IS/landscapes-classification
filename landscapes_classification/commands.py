import sys
from pathlib import Path
from typing import List

import fire
from hydra import compose, initialize_config_dir

from landscapes_classification.command_funcs.infer import infer_command
from landscapes_classification.command_funcs.train import train_command


class Commands:
    """
    Entrypoint class for train and inference functions
    """

    def __init__(self, overrides: List[str]):
        self.overrides = overrides

    def train(self):
        with initialize_config_dir(
            version_base=None, config_dir=str(Path(__file__).parents[1] / "config")
        ):
            cfg = compose(config_name="config", overrides=self.overrides)
            train_command(cfg)

    def infer(self):
        with initialize_config_dir(
            version_base=None, config_dir=str(Path(__file__).parents[1] / "config")
        ):
            for argv in sys.argv[2:]:
                sys.argv.remove(argv)
            cfg = compose(config_name="config", overrides=self.overrides)
            infer_command(cfg)


def main():
    overrides = sys.argv[2:] if len(sys.argv) > 2 else []
    sys.argv = sys.argv[:2]
    fire.Fire(Commands(overrides))


if __name__ == "__main__":
    main()
