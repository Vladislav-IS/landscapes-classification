import subprocess
from typing import Any

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def dvc_pull() -> None:
    print("Data folder not found. Downloading data from the S3 storage...")
    try:
        subprocess.run(["poetry", "run", "dvc", "pull"], check=True, text=True, capture_output=True)
        return True
    except Exception as e:
        print(f"Critical: DVC pull failed with error: {str(e)}")
        return False


def init_dataset(path: str, mode: str, size: int) -> ImageFolder:
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


def init_dataloader(
    dataset: Any, batch_size: int, shuffle: bool = True, num_workers: int = 4
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
    )
