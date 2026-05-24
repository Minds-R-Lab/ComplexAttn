"""Shortest-path-through-noisy-graph classification task.

Each example:
  - A randomly weighted 8x8 directed graph (weights in [0.1, 1.0])
  - Some "non-edges" are set to a high weight (5.0) with probability 0.3
  - Input: flattened 64-dim adjacency-with-weights matrix
  - Label: shortest-path distance from node 0 to node 7, bucketed into
    4 quantile-balanced classes

The task is designed so a min-plus (tropical) layer is *literally* one
Bellman-Ford iteration: D'[i,j] = min_k (D[i,k] + D[k,j]). An L-layer
min-plus network with L >= 8 can compute exact shortest paths.

An MLP must learn this combinatorial structure from scratch with no
native primitive for min-plus matrix multiplication.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def floyd_warshall(W: np.ndarray) -> np.ndarray:
    """Standard min-plus matrix-matrix iteration."""
    n = W.shape[0]
    D = W.copy().astype(np.float64)
    for k in range(n):
        D = np.minimum(D, D[:, k:k+1] + D[k:k+1, :])
    return D


def make_graphs(n_samples: int, N: int = 8, density: float = 0.7, seed: int = 0):
    rng = np.random.default_rng(seed)
    Xs, dists = [], []
    big = 5.0  # weight assigned to "non-edges"
    # Oversample, then trim to n_samples (some graphs are disconnected → drop).
    target = n_samples
    iters = 0
    while len(Xs) < target and iters < target * 5:
        iters += 1
        W = rng.uniform(0.1, 1.0, (N, N)).astype(np.float32)
        mask = rng.uniform(0, 1, (N, N)) < density
        W = np.where(mask, W, big).astype(np.float32)
        np.fill_diagonal(W, 0.0)
        D = floyd_warshall(W)
        d = D[0, N - 1]
        if d >= big:  # 0→N-1 effectively unreachable; skip
            continue
        Xs.append(W.flatten())
        dists.append(d)
    X = np.asarray(Xs, dtype=np.float32)
    dists = np.asarray(dists, dtype=np.float32)
    return X, dists


def make_dataset(n_per_class: int = 750, N: int = 8, seed: int = 0):
    """Returns (X, y) with quantile-balanced 4 classes."""
    X, dists = make_graphs(n_samples=n_per_class * 4, N=N, seed=seed)
    # quantile-balanced bucketing
    qs = np.quantile(dists, [0.25, 0.5, 0.75])
    y = np.digitize(dists, qs).astype(np.int64)
    # shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_loaders(n_train: int = 2000, n_val: int = 500, batch_size: int = 64, seed: int = 0):
    total = n_train + n_val
    X, y = make_dataset(n_per_class=total // 4 + 1, seed=seed)
    X, y = X[:total], y[:total]
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
    print(f"X: {X.shape} dtype={X.dtype}")
    print(f"y: {y.shape}  per-class counts: {np.bincount(y)}")
    print(f"sample distances range: ~{X.mean():.2f} +/- {X.std():.2f}")
