"""MNIST data loading. Uses torchvision; downloads to ./data on first run."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist_loaders(
    data_root: str | Path = "./data",
    batch_size: int = 128,
    train_subset: int | None = None,
    val_subset: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(str(data_root), train=True, download=True, transform=transform)
    val_ds = datasets.MNIST(str(data_root), train=False, download=True, transform=transform)

    if train_subset is not None and train_subset < len(train_ds):
        train_ds = Subset(train_ds, list(range(train_subset)))
    if val_subset is not None and val_subset < len(val_ds):
        val_ds = Subset(val_ds, list(range(val_subset)))

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    return train_loader, val_loader
