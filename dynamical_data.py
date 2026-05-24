"""Synthetic dynamical-system trajectory classification.

Four classes with distinct frequency-domain signatures:
    0: pure sinusoid          A sin(omega t + phi)
    1: damped sinusoid        A exp(-gamma t) sin(omega t + phi)
    2: two-tone (beats)       sin(w1 t) + sin(w2 t)  with w1 ~ w2
    3: linear chirp           sin((w0 + alpha t) t + phi)

All trajectories are T-step 1D signals with additive Gaussian noise.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_dataset(
    n_per_class: int = 625,
    T: int = 64,
    dt: float = 0.1,
    noise: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (X, y). X has shape (N, T) float32, y has shape (N,) int64."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) * dt
    Xs, ys = [], []
    for cls in range(4):
        for _ in range(n_per_class):
            phi = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.7, 1.3)
            if cls == 0:
                w = rng.uniform(1.5, 5.0)
                x = amp * np.sin(w * t + phi)
            elif cls == 1:
                w = rng.uniform(1.5, 5.0)
                gamma = rng.uniform(0.08, 0.25)
                x = amp * np.exp(-gamma * t) * np.sin(w * t + phi)
            elif cls == 2:
                w1 = rng.uniform(1.5, 4.5)
                w2 = w1 + rng.uniform(0.3, 0.7)
                x = amp * (np.sin(w1 * t + phi) + np.sin(w2 * t)) / 2.0
            else:  # cls == 3 — chirp
                w0 = rng.uniform(1.0, 2.0)
                alpha = rng.uniform(0.3, 0.7)
                x = amp * np.sin((w0 + alpha * t) * t + phi)
            x = x + noise * rng.normal(size=T)
            Xs.append(x)
            ys.append(cls)
    X = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_loaders(
    n_train: int = 2000,
    n_val: int = 500,
    T: int = 64,
    dt: float = 0.1,
    noise: float = 0.03,
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    total_needed = n_train + n_val
    per_class = total_needed // 4 + 1
    X, y = make_dataset(n_per_class=per_class, T=T, dt=dt, noise=noise, seed=seed)
    X = X[:total_needed]; y = y[:total_needed]
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:], y[n_train:]
    g = torch.Generator(); g.manual_seed(seed)
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                    batch_size=batch_size, shuffle=False, drop_last=False)
    return tr, va


if __name__ == "__main__":
    X, y = make_dataset(n_per_class=10, seed=0)
    print(f"X: {X.shape} {X.dtype}, y: {y.shape}")
    print(f"per-class counts: {np.bincount(y)}")
