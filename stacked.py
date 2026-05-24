"""Stacked RMC and stacked MLP for the depth experiment.

A StackedRMC has L Resonant Manifold Cells in sequence. Each cell:
  - takes (x_0, p_0) on its manifold
  - integrates a Hamiltonian flow (reversible, symplectic)
  - emits K resonant-readout features

Between cells, a small linear map projects the K features back to
(x_0, p_0) for the next cell. The resonant readout plays the same
architectural role between layers that ReLU plays between MLP layers.

The MLP baseline (DeepMLP) has L hidden layers of width h, matched in
parameter count to the corresponding RMC depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import ResonantManifoldCell


class StackedRMC(nn.Module):
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
        freeze_B: bool = True,  # use the simplified RMC (no quadratic potential)
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.encoder = nn.Linear(input_dim, 2 * manifold_dim)

        self.cells = nn.ModuleList([
            ResonantManifoldCell(
                manifold_dim=manifold_dim, num_modes=num_modes,
                num_steps=num_steps, dt=dt, potential_hidden=potential_hidden,
            ) for _ in range(num_layers)
        ])
        # Inter-layer projection: K resonant features -> (x_0, p_0) for next cell.
        self.interlayer = nn.ModuleList([
            nn.Linear(num_modes, 2 * manifold_dim)
            for _ in range(num_layers - 1)
        ])
        self.head = nn.Linear(num_modes, num_classes)

        if freeze_B:
            for cell in self.cells:
                with torch.no_grad():
                    cell.B_raw.zero_()
                cell.B_raw.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        enc = self.encoder(x)
        x0, p0 = enc.chunk(2, dim=-1)
        for i, cell in enumerate(self.cells):
            features = cell(x0, p0)  # (B, K)
            if i < self.num_layers - 1:
                next_enc = self.interlayer[i](features)
                x0, p0 = next_enc.chunk(2, dim=-1)
        return self.head(features)


class DeepMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        num_layers: int = 2,
        hidden_dim: int = 32,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Param counts at n=3000 dynamical setup:")
    print(f"{'L':>3s}  {'RMC':>8s}  {'MLP-32':>8s}")
    for L in [1, 2, 4]:
        r = StackedRMC(num_layers=L)
        m = DeepMLP(num_layers=L)
        print(f"{L:>3d}  {count_trainable(r):>8d}  {count_trainable(m):>8d}")

    # quick forward sanity check
    x = torch.randn(4, 64)
    for L in [1, 2, 4]:
        out = StackedRMC(num_layers=L)(x)
        print(f"L={L}: StackedRMC out shape {out.shape}")
