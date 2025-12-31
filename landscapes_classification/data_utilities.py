import subprocess
import time
from typing import Any, List

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def dvc_pull(max_conn_tries) -> bool:
    """
    Pull the data from remote repository

    :param conn_tries: Maximum number of connection tries
    :type path: int
    :return: Flag of success
    :rtype: bool
    """
    print("Info: Data folder not found. Downloading data from the S3 storage...")
    result = None
    for try_num in range(max_conn_tries):
        print(f"Info: Trial {try_num}...")
        result = subprocess.run(["dvc", "pull"], capture_output=True, text=True)
        if result.returncode == 0:
            print("Info: DVC data pulled successfully")
            return True
        if try_num < max_conn_tries - 1:
            time.sleep(10)
    print(f"Critical: DVC pull failed with error: {result.stderr}")
    return False


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
