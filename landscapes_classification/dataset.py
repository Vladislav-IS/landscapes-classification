from pathlib import Path
from typing import List

import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LandscapesDataset(Dataset):
    def __init__(self, files: List[Path], labels: List[int], mode: str, size: int):
        super().__init__()
        self.files = files
        self.labels = labels
        self.mode = mode
        DATA_MODES = ["train", "val", "test"]
        if self.mode not in DATA_MODES:
            raise NameError(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
        if self.mode == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")
        return self.transform(image), self.labels[index]


class LandscapesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, size: int, seed: int = 42):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.size = size

    def prepare_data(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

    def setup(self):
        label_names = sorted(path for path in self.data_dir.iterdir() if path.is_dir())
        label_to_idx = {name.name: idx for idx, name in enumerate(label_names)}
        files = list(self.data_dir.rglob("*.jpg"))
        labels = [label_to_idx[file.parent.name] for file in files]
        train_files, val_files, train_labels, val_labels = train_test_split(
            files, labels, test_size=0.25, stratify=labels, random_state=self.seed
        )
        val_files, test_files, val_labels, test_labels = train_test_split(
            val_files, val_labels, test_size=0.5, stratify=val_labels, random_state=self.seed
        )
        self.train_dataset = LandscapesDataset(
            train_files, train_labels, mode="train", size=self.size
        )
        self.val_dataset = LandscapesDataset(val_files, val_labels, mode="val", size=self.size)
        self.test_dataset = LandscapesDataset(test_files, test_labels, mode="test", size=self.size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
