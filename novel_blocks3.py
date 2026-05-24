"""Round 3: order-statistic and contrast-based blocks. These are mostly
non-polynomial and structurally distinct from Bilinear/GLU.

  1. PairwiseDiffBlock: for each output, sum learnable-weighted pairwise
     |a_i - a_j| over a small set of features. Genuinely novel — computes a
     learned 'distance matrix' summary.
  2. TropMaxBlock:  y = x + W_out · max(W_a x, W_b x)  (tropical OR)
  3. TropMinBlock:  y = x + W_out · min(W_a x, W_b x)  (tropical AND)
  4. MultiAnchorBlock: y = x + W_out · sum_j |W_a x - c_j|, learnable anchors
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PairwiseDiffBlock(nn.Module):
    """For each output, aggregate pairwise |a_i - a_j| of a small projection."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.branch = nn.Linear(dim, hidden_dim)
        # Output mixes the n*(n-1)/2 pairwise distances
        n_pairs = hidden_dim * (hidden_dim - 1) // 2
        self.out = nn.Linear(n_pairs, dim, bias=True)
        # Pre-compute index pairs (i, j) with i < j
        i, j = torch.triu_indices(hidden_dim, hidden_dim, offset=1)
        self.register_buffer("idx_i", i)
        self.register_buffer("idx_j", j)
        with torch.no_grad(): self.out.weight.mul_(0.5)

    def forward(self, x):
        h = self.branch(x)                                          # (B, h)
        pairwise = torch.abs(h[:, self.idx_i] - h[:, self.idx_j])    # (B, n_pairs)
        return x + self.out(pairwise)


class TropMaxBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.branch_a = nn.Linear(dim, hidden_dim)
        self.branch_b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        with torch.no_grad(): self.out.weight.mul_(0.5)
    def forward(self, x):
        return x + self.out(torch.maximum(self.branch_a(x), self.branch_b(x)))


class TropMinBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.branch_a = nn.Linear(dim, hidden_dim)
        self.branch_b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        with torch.no_grad(): self.out.weight.mul_(0.5)
    def forward(self, x):
        return x + self.out(torch.minimum(self.branch_a(x), self.branch_b(x)))


class MultiAnchorBlock(nn.Module):
    """y_i = x_i + sum_j W_ij · sum_d |(W_a x)_d - c_{j,d}|.
    Each output reads from multiple distances to learnable anchor points."""
    def __init__(self, dim, hidden_dim, num_anchors=8):
        super().__init__()
        self.branch = nn.Linear(dim, hidden_dim)
        self.anchors = nn.Parameter(torch.randn(num_anchors, hidden_dim) * 0.5)
        # Each anchor produces a scalar (mean of |diffs|), and we mix anchor-scalars to dim
        self.out = nn.Linear(num_anchors, dim)
        with torch.no_grad(): self.out.weight.mul_(0.5)
    def forward(self, x):
        h = self.branch(x).unsqueeze(1)              # (B, 1, hidden)
        d = torch.abs(h - self.anchors.unsqueeze(0)) # (B, n_anchors, hidden)
        d_summary = d.mean(dim=-1)                   # (B, n_anchors)
        return x + self.out(d_summary)


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


def pairwise_net(): return GenericNet(lambda d: PairwiseDiffBlock(d, 10))
def tropmax_net():  return GenericNet(lambda d: TropMaxBlock(d, 16))
def tropmin_net():  return GenericNet(lambda d: TropMinBlock(d, 16))
def anchor_net():   return GenericNet(lambda d: MultiAnchorBlock(d, 16, num_anchors=10))


if __name__ == "__main__":
    from stacked import DeepMLP
    print(f"{'block':>14s}  {'L=2 params':>12s}  {'MLP-L2':>9s}")
    for name, fac in [("PairwiseDiff", pairwise_net), ("TropMax", tropmax_net),
                      ("TropMin", tropmin_net), ("MultiAnchor", anchor_net)]:
        print(f"{name:>14s}  {count_p(fac()):>12d}  {count_p(DeepMLP(num_layers=2)):>9d}")
