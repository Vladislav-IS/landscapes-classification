from typing import Any, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def init_dataset(path: str, mode: str, size: int):
    if mode == "train":
        transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize(size),
                transforms.ToTensor(),
            ]
        )
    else:
        transformer = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    return ImageFolder(path, transformer)


def init_dataloader(dataset: Any, batch_size: int, shuffle: bool = True, num_workers: int = 4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


class LandscapesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        batch_size: int,
        num_workers: int,
        size: int,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = init_dataset(self.train_dir, "train", self.size)
        self.val_dataset = init_dataset(self.val_dir, "val", self.size)
        self.test_dataset = init_dataset(self.test_dir, "test", self.size)

    def train_dataloader(self):
        return init_dataloader(self.train_dataset, self.batch_size)

    def val_dataloader(self):
        return init_dataloader(self.val_dataset, self.batch_size, shuffle=False)

    def test_dataloader(self):
        return init_dataloader(self.test_dataset, self.batch_size, shuffle=False)
