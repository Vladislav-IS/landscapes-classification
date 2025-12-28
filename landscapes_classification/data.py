import shutil
from pathlib import Path
from typing import Any

import dvc.api
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def dvc_pull(repo_url: str, data_path: str, local_dir: str, rev: str) -> bool:
    local_path = Path(local_dir)
    if local_path.rglob("*.jpg"):
        return False
    local_path.mkdir(exist_ok=True)
    files = dvc.api.read_dir(path=data_path, repo=repo_url, rev=rev)
    if not files:
        return False
    for file_path in files:
        full_path = f"{data_path}/{file_path}"
        local_file = local_path / file_path
        local_file.parent.mkdir(parents=True, exist_ok=True)
        with dvc.api.open(full_path, repo=repo_url, rev=rev, mode="rb") as f_remote:
            with open(local_file, "wb") as f_local:
                shutil.copyfileobj(f_remote, f_local)
    return True


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
    )
