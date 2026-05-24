"""TropicalNet — networks built on the (max, +) tropical semiring instead of
the standard (+, x) ring.

A tropical "linear" layer computes
    y_i = max_j (W^+_ij + x_j) - max_j (W^-_ij + x_j) + b_i
This is a continuous piecewise-linear function (a tropical polynomial), and
two such layers can together represent any continuous piecewise-linear
function R^n -> R^m (Zhang, Naitzat, Lim 2018). The "nonlinearity" is built
into max — no separate ReLU is needed.

Why this is a fundamentally different primitive from ReLU+linear:
- ReLU networks partition input space along *hyperplanes* defined by
  Σ_j w_j x_j + b = 0. Tropical networks partition along *max-witnesses*:
  the regions where a particular j wins the max.
- Same function class (CPWL), different decision-boundary geometry.
- Should beat ReLU networks on tasks where the natural structure is
  combinatorial / shortest-path / max-flow rather than additive.

Implementation note: max() has sparse gradient (only the argmax-j gets it).
We mitigate two ways:
- Larger weight init (std=1) so W contributes meaningfully to which j wins;
  otherwise x dominates and W gets no signal.
- The (W^+, W^-) parameterization gives two paths per output, doubling the
  effective gradient routing per layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TropicalLayer(nn.Module):
    """y_i = max_j(W+_ij + x_j) - max_j(W-_ij + x_j) + b_i  (tropical polynomial)."""

    def __init__(self, in_dim: int, out_dim: int, init_scale: float = 1.0) -> None:
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(out_dim, in_dim) * init_scale)
        self.W_neg = nn.Parameter(torch.randn(out_dim, in_dim) * init_scale)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (B, in_dim)
        # W+:   (out_dim, in_dim)
        # Need max_j (W_ij + x_j) for each (B, i).
        # Broadcast: x.unsqueeze(1) is (B, 1, in_dim); W.unsqueeze(0) is (1, out_dim, in_dim).
        # Sum is (B, out_dim, in_dim), max over last dim gives (B, out_dim).
        max_pos = (x.unsqueeze(1) + self.W_pos.unsqueeze(0)).max(dim=-1).values
        max_neg = (x.unsqueeze(1) + self.W_neg.unsqueeze(0)).max(dim=-1).values
        return max_pos - max_neg + self.b


class TropicalNet(nn.Module):
    """L hidden tropical layers + a linear output head. Structurally mirrors
    the MLP baseline (L hidden Linear+ReLU + 1 output Linear)."""

    def __init__(
        self,
        input_dim: int = 64,
        num_layers: int = 2,
        hidden_dim: int = 20,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [TropicalLayer(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(TropicalLayer(hidden_dim, hidden_dim))
        self.hidden_layers = nn.ModuleList(layers)
        # Linear output head — keeps the classifier head consistent with MLP.
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.head(x)


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    from stacked import DeepMLP
    print(f"{'L':>3s}  {'TropicalNet':>12s}  {'DeepMLP-32':>12s}")
    for L in [1, 2, 4]:
        t = TropicalNet(num_layers=L)
        m = DeepMLP(num_layers=L)
        print(f"{L:>3d}  {count_trainable(t):>12d}  {count_trainable(m):>12d}")
    # Forward + gradient sanity at L=4
    t = TropicalNet(num_layers=4); t.train()
    x = torch.randn(4, 64); y = torch.randint(0, 4, (4,))
    loss = torch.nn.functional.cross_entropy(t(x), y); loss.backward()
    nz = sum(1 for p in t.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    tot = sum(1 for p in t.parameters() if p.requires_grad)
    print(f"\nL=4 gradient sanity: {nz}/{tot} params received nonzero gradient, loss={loss.item():.4f}")

class MinPlusLayer(nn.Module):
    """y_i = min_j(W_ij + x_j) + b_i  (one Bellman-Ford iteration if W is
    interpreted as adjacency)."""

    def __init__(self, in_dim: int, out_dim: int, init_scale: float = 1.0) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * init_scale)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.unsqueeze(1) + self.W.unsqueeze(0)).min(dim=-1).values + self.b


class MinPlusNet(nn.Module):
    """L hidden min-plus layers + a linear output head. Each min-plus layer
    can implement one Bellman-Ford iteration over an N-node graph."""

    def __init__(
        self,
        input_dim: int = 64,
        num_layers: int = 2,
        hidden_dim: int = 20,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [MinPlusLayer(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(MinPlusLayer(hidden_dim, hidden_dim))
        self.hidden_layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.head(x)

