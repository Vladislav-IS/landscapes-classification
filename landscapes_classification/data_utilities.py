import subprocess
from pathlib import Path
from typing import Any, List

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def dvc_pull() -> None:
    """
    Pull the data from remote repository

    :return:
    :rtype: bool
    """
    print("Info: Data folder not found. Downloading data from the S3 storage...")
    repo_path = Path(__file__).parents[1]
    result = subprocess.run(["dvc", "pull"], capture_output=True, text=True, cwd=str(repo_path))
    if result.returncode != 0:
        print(f"Critical: DVC pull failed with error: {result.stderr}")
        return False
    print("Info: DVC data pulled successfully")
    return True


def init_dataset(
    path: str, mode: str, size: int, mean: List[float], std: List[float]
) -> ImageFolder:
    """
    Create dataset from raw data

    :param path: Path to data folder
    :type path: str
    :param mode: Mode of dataset (train/val/test)
    :type mode: str
    :param size: Size of image after preprocessing
    :type size: int
    :param mean: Means by channel for image normalization
    :type mean: List[float]
    :param std: Stds by channel for image normalization
    :type std: List[float]
    :return: Prepared PyTorch Dataset
    :rtype: ImageFolder
    """
    if mode == "train":
        transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transformer = transforms.Compose(
            [transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    return ImageFolder(path, transformer)


def init_dataloader(
    dataset: Any, batch_size: int, shuffle: bool = True, num_workers: int = 4
) -> DataLoader:
    """
    Create dataloader from prepared dataset

    :param dataset: Prepared Dataset
    :type dataset: Any
    :param batch_size: Size of batch
    :type batch_size: int
    :param shuffle: Flag of shuffling the data
    :type shuffle: bool
    :param num_workers: Workers number
    :type num_workers: int
    :return: Prepared PyTorch Dataloader
    :rtype: DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
    )
