"""Multiplicative Residual Block (MRB).

The block computes y = x + W_out @ ( (W_a @ x) * (W_b @ x) )
where * is element-wise. This introduces multiplicative feature interactions
on top of a residual linear stream — the network can represent polynomial
(specifically: at depth L, polynomials of total degree up to 2^L) features
of the input while keeping smooth dense gradients and a clean residual path.

Three properties this is meant to combine that the previous primitives lacked:
- Dense smooth gradients (no max/min/softmax)
- Multiplicative feature interactions (richer than purely additive ReLU)
- Residual composition (no information bottleneck under depth)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MRB(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.branch_a = nn.Linear(dim, hidden_dim)
        self.branch_b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        with torch.no_grad():
            # Slight scale-down on the output so the residual block starts
            # close to identity (but not zero — we want gradient flow).
            self.out.weight.mul_(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch_a(x)
        b = self.branch_b(x)
        return x + self.out(a * b)


class MRBNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        num_layers: int = 2,
        dim: int = 20,
        hidden_dim: int = 16,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([MRB(dim, hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x.view(x.size(0), -1))
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    from stacked import DeepMLP
    print(f"{'L':>3s}  {'MRBNet':>8s}  {'DeepMLP':>8s}")
    for L in [1, 2, 4]:
        m = MRBNet(num_layers=L)
        mlp = DeepMLP(num_layers=L)
        print(f"{L:>3d}  {count_trainable(m):>8d}  {count_trainable(mlp):>8d}")
    # Forward + gradient sanity
    m = MRBNet(num_layers=4); m.train()
    x = torch.randn(4, 64); y = torch.randint(0, 4, (4,))
    loss = torch.nn.functional.cross_entropy(m(x), y); loss.backward()
    nz = sum(1 for p in m.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    tot = sum(1 for p in m.parameters() if p.requires_grad)
    print(f"\nL=4 gradient sanity: {nz}/{tot} params received nonzero gradient")
