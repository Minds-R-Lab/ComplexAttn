"""More candidate building blocks, designed to be structurally distinct from
Bilinear/GLU and from each other. Each one uses a *non-polynomial* primitive."""

from __future__ import annotations

import torch
import torch.nn as nn


class SineBlock(nn.Module):
    """y = x + W_out · sin(W_a x + b).  Periodic activation, SIREN-style."""
    def __init__(self, dim, hidden_dim, omega=3.0):
        super().__init__()
        self.branch = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.omega = omega
        with torch.no_grad():
            # SIREN init: scale first-layer weights by 1/in so sin doesn't saturate
            self.branch.weight.mul_(1.0 / dim**0.5)
            self.out.weight.mul_(0.5)

    def forward(self, x):
        return x + self.out(torch.sin(self.omega * self.branch(x)))


class ComparatorXORBlock(nn.Module):
    """y = x + W_out · (1 - tanh(W_a x) · tanh(W_b x)) / 2.  Soft XOR — high when
    the two projections disagree in sign."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.branch_a = nn.Linear(dim, hidden_dim)
        self.branch_b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        with torch.no_grad(): self.out.weight.mul_(0.5)

    def forward(self, x):
        a = torch.tanh(self.branch_a(x))
        b = torch.tanh(self.branch_b(x))
        return x + self.out((1.0 - a * b) * 0.5)


class TopKBlock(nn.Module):
    """y = x + W_out · top_k(W_a x).  Hard sparsity — keep only top-k activations."""
    def __init__(self, dim, hidden_dim, k_frac=0.25):
        super().__init__()
        self.branch = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.k = max(1, int(hidden_dim * k_frac))
        with torch.no_grad(): self.out.weight.mul_(0.5)

    def forward(self, x):
        h = self.branch(x)
        # mask all but top-k per row
        topk_vals, topk_idx = h.topk(self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        return x + self.out(h * mask)


class RangeBlock(nn.Module):
    """y = x + W_out · (max(W x reshaped into groups) - min(same)). Uses range of
    grouped features as an order-statistic feature."""
    def __init__(self, dim, hidden_dim, group_size=4):
        super().__init__()
        self.group_size = group_size
        # ensure hidden_dim divisible by group_size
        self.hidden_dim = (hidden_dim // group_size) * group_size
        self.n_groups = self.hidden_dim // group_size
        self.branch = nn.Linear(dim, self.hidden_dim)
        self.out = nn.Linear(self.n_groups, dim)
        with torch.no_grad(): self.out.weight.mul_(0.5)

    def forward(self, x):
        h = self.branch(x)                                            # (B, hidden)
        h = h.view(h.size(0), self.n_groups, self.group_size)        # (B, G, k)
        rng = h.max(dim=-1).values - h.min(dim=-1).values             # (B, G)
        return x + self.out(rng)


class GenericNet(nn.Module):
    def __init__(self, block_factory, input_dim=64, num_layers=2, dim=20, num_classes=4):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([block_factory(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, num_classes)
    def forward(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        for b in self.blocks: h = b(h)
        return self.head(h)


def count_p(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def sine_net(num_layers=2):  return GenericNet(lambda d: SineBlock(d, 16), num_layers=num_layers)
def xor_net(num_layers=2):   return GenericNet(lambda d: ComparatorXORBlock(d, 16), num_layers=num_layers)
def topk_net(num_layers=2):  return GenericNet(lambda d: TopKBlock(d, 16, k_frac=0.25), num_layers=num_layers)
def range_net(num_layers=2): return GenericNet(lambda d: RangeBlock(d, 16, group_size=4), num_layers=num_layers)


if __name__ == "__main__":
    from stacked import DeepMLP
    print(f"{'block':>12s}  {'L=2 params':>12s}  {'MLP-L2':>9s}")
    for name, fac in [("Sine", sine_net), ("ComparatorXOR", xor_net),
                      ("TopK", topk_net), ("Range", range_net)]:
        print(f"{name:>12s}  {count_p(fac(num_layers=2)):>12d}  {count_p(DeepMLP(num_layers=2)):>9d}")
