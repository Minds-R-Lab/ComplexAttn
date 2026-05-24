"""ModReLUStackedRMC — depth via a phase-equivariant inter-cell nonlinearity.

Design rationale from the previous failed redesigns:
- StackedRMC (|a_k| between cells) failed because the magnitude readout
  destroys phase information that the next cell's flow needs.
- ResidualRMC + phase-preserving (Re, Im) failed because preserving phase
  destroyed the architecture's main useful inductive bias (phase invariance
  for classification).

modReLU is a phase-EQUIVARIANT nonlinearity from complex-valued NNs:
    f(z) = ReLU(|z| + b) * z / |z|
It treats (x, p) as a complex vector z = x + i*p and applies a magnitude-
thresholded gating while preserving phase. So:
  - Each cell still does a Hamiltonian flow (information-preserving inside)
  - Between cells, modReLU provides selectivity without destroying phase
  - Final readout is |a_k| — phase-invariant for classification

This is the only redesign that simultaneously preserves phase between cells
(so flows compose) and gives phase-invariant final features (for the
classifier).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import ResonantManifoldCell


class ModReLU(nn.Module):
    """Phase-equivariant nonlinearity: f(x, p) = ReLU(|z| + b) * z / |z|
    where z = x + i*p. Returns (x_out, p_out).

    Bias b is per-dimension and is initialized slightly negative so the
    nonlinearity creates an actual threshold at init."""

    def __init__(self, dim: int, init_bias: float = -0.05):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), init_bias))

    def forward(self, x: torch.Tensor, p: torch.Tensor):
        # Complex magnitude: |z| = sqrt(x^2 + p^2)
        mag = torch.sqrt(x.pow(2) + p.pow(2) + 1e-8)
        gate = torch.relu(mag + self.bias) / mag    # >= 0, equals 0 where |z| < -b
        return x * gate, p * gate


class ModReLUStackedRMC(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        num_layers: int = 2,
        manifold_dim: int = 12,
        num_modes: int = 24,
        num_classes: int = 4,
        num_steps: int = 12,
        dt: float = 0.15,
        potential_hidden: int = 16,
        freeze_B: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.encoder = nn.Linear(input_dim, 2 * manifold_dim)
        self.cells = nn.ModuleList([
            ResonantManifoldCell(manifold_dim, num_modes, num_steps, dt, potential_hidden)
            for _ in range(num_layers)
        ])
        if freeze_B:
            for cell in self.cells:
                with torch.no_grad(): cell.B_raw.zero_()
                cell.B_raw.requires_grad = False
        # modReLU between layers, applied to the phase-space state
        self.modrelus = nn.ModuleList([
            ModReLU(manifold_dim) for _ in range(num_layers - 1)
        ])
        # Small linear "remix" on (x, p) between layers — gives the next cell's
        # flow some freedom to repurpose the upstream state. Without this the
        # stack is just a longer integration of essentially one Hamiltonian.
        self.remix = nn.ModuleList([
            nn.Linear(2 * manifold_dim, 2 * manifold_dim) for _ in range(num_layers - 1)
        ])
        self.head = nn.Linear(num_modes, num_classes)

    def forward(self, x: torch.Tensor):
        enc = self.encoder(x.view(x.size(0), -1))
        x_cur, p_cur = enc.chunk(2, dim=-1)
        for i in range(self.num_layers):
            cell = self.cells[i]
            if i < self.num_layers - 1:
                # Integrate, take final (x_T, p_T), apply modReLU + remix, hand off
                xs, ps = cell.integrate_with_trajectory(x_cur, p_cur)
                x_T, p_T = xs[-1], ps[-1]
                x_T, p_T = self.modrelus[i](x_T, p_T)            # phase-equivariant gate
                remix = self.remix[i](torch.cat([x_T, p_T], dim=-1))
                x_cur, p_cur = remix.chunk(2, dim=-1)
            else:
                # Final cell: do the proper resonant readout
                features = cell(x_cur, p_cur)
        return self.head(features)


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    from stacked import DeepMLP
    print(f"{'L':>3s}  {'ModReLU-RMC':>12s}  {'DeepMLP-32':>12s}")
    for L in [1, 2, 4]:
        r = ModReLUStackedRMC(num_layers=L)
        m = DeepMLP(num_layers=L)
        print(f"{L:>3d}  {count_trainable(r):>12d}  {count_trainable(m):>12d}")
    # Forward + gradient sanity at L=4
    m = ModReLUStackedRMC(num_layers=4); m.train()
    x = torch.randn(4, 64); y = torch.randint(0, 4, (4,))
    loss = torch.nn.functional.cross_entropy(m(x), y); loss.backward()
    nz = sum(1 for p in m.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    tot = sum(1 for p in m.parameters() if p.requires_grad)
    print(f"\nL=4 gradient sanity: {nz}/{tot} params received nonzero gradient")
