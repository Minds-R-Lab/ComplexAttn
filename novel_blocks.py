"""Candidate building blocks beyond Bilinear/GLU.

Each block has matched-param sizing for L=2 on the dynamical task.
We compare three structurally distinct primitives:

  1. AbsDiff:   y = x + W_out @ |W_a x - W_b x|       (contrast detector)
  2. Antisymm:  y = x + (W - W^T) @ tanh(x)            (volume-preserving feedforward)
  3. Cubic:     y = x + W_out @ (W_a x)^3              (degree-3 polynomial single-projection)
  4. SquaredDiff: y = x + W_out @ (W_a x - W_b x)^2   (smooth squared contrast)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AbsDiffBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.branch_a = nn.Linear(dim, hidden_dim)
        self.branch_b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        with torch.no_grad(): self.out.weight.mul_(0.5)

    def forward(self, x):
        return x + self.out(torch.abs(self.branch_a(x) - self.branch_b(x)))


class AntisymBlock(nn.Module):
    """Antisymmetric residual feedforward, after Chang et al. 2018 AntisymRNN."""
    def __init__(self, dim, gamma=0.3):
        super().__init__()
        # We use a single linear layer's weight; antisymmetric part = (W - W^T)/2
        self.W = nn.Parameter(torch.randn(dim, dim) / dim**0.5)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.gamma = gamma

    def forward(self, x):
        A = self.W - self.W.T  # antisymmetric part
        return x + self.gamma * torch.tanh(x @ A.T + self.bias)


class CubicBlock(nn.Module):
    """y = x + W_out @ (W_a x)^3.  Degree-3 polynomial features from one projection."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.branch = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        with torch.no_grad(): self.out.weight.mul_(0.5)

    def forward(self, x):
        a = self.branch(x)
        return x + self.out(a * a * a)


class SquaredDiffBlock(nn.Module):
    """y = x + W_out @ (W_a x - W_b x)^2.  Smooth squared contrast."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.branch_a = nn.Linear(dim, hidden_dim)
        self.branch_b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        with torch.no_grad(): self.out.weight.mul_(0.5)

    def forward(self, x):
        d = self.branch_a(x) - self.branch_b(x)
        return x + self.out(d * d)


class GenericBlockNet(nn.Module):
    """Encoder → L residual blocks → linear head. Architecture parametric over block."""

    def __init__(self, block_factory, input_dim=64, num_layers=2,
                 dim=20, num_classes=4):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([block_factory(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# Block factories sized so that L=2 net ~ MLP-L2's 3268 params.
def absdiff_net(num_layers=2): return GenericBlockNet(lambda d: AbsDiffBlock(d, 16), num_layers=num_layers)
def antisym_net(num_layers=2): return GenericBlockNet(lambda d: AntisymBlock(d), num_layers=num_layers)
def cubic_net(num_layers=2):   return GenericBlockNet(lambda d: CubicBlock(d, 16), num_layers=num_layers)
def sqdiff_net(num_layers=2):  return GenericBlockNet(lambda d: SquaredDiffBlock(d, 16), num_layers=num_layers)


def count_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    from stacked import DeepMLP
    print(f"{'block':>10s}  {'L=2 params':>12s}  {'MLP-L2':>9s}")
    for name, fac in [("AbsDiff", absdiff_net), ("Antisym", antisym_net),
                      ("Cubic", cubic_net), ("SqDiff", sqdiff_net)]:
        m = fac(num_layers=2)
        print(f"{name:>10s}  {count_trainable(m):>12d}  {count_trainable(DeepMLP(num_layers=2)):>9d}")
