"""CoupledOscillatorRMC — embraces what we learned about the architecture.

After ResidualRMC failed (phase-preserving readout broke the architecture's
phase-invariance, which is exactly what makes |a_k| a useful classifier
feature), we know the architecture is fundamentally a *one-shot* phase-
invariant spectral feature extractor that resists stacking.

This design plays to that: instead of stacking layers, it makes ONE cell
much richer:
  - d=24 manifold (vs d=12 originally) — twice as many oscillators
  - K=24 resonant modes
  - T=20 leapfrog steps (vs 12) — more time for couplings to act
  - B kept LEARNABLE (no longer frozen) — B is literally the coupling
    matrix between oscillators, so we want it adaptive

Param count is matched to the 4-layer MLP-32 baseline (~5400).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import ResonantManifoldCell


class CoupledOscillatorRMC(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        manifold_dim: int = 24,
        num_modes: int = 24,
        num_classes: int = 4,
        num_steps: int = 20,
        dt: float = 0.1,
        potential_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, 2 * manifold_dim)
        self.flow = ResonantManifoldCell(
            manifold_dim=manifold_dim,
            num_modes=num_modes,
            num_steps=num_steps,
            dt=dt,
            potential_hidden=potential_hidden,
        )
        # B is LEARNABLE here — it's the coupling matrix between oscillators.
        self.head = nn.Linear(num_modes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x.view(x.size(0), -1))
        x0, p0 = enc.chunk(2, dim=-1)
        features = self.flow(x0, p0)         # phase-invariant |a_k| readout
        return self.head(features)


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    from stacked import DeepMLP
    co = CoupledOscillatorRMC()
    mlp_l1 = DeepMLP(num_layers=1)
    mlp_l4 = DeepMLP(num_layers=4)
    print(f"CoupledOscillatorRMC params: {count_trainable(co)}")
    print(f"DeepMLP L=1: {count_trainable(mlp_l1)}")
    print(f"DeepMLP L=4: {count_trainable(mlp_l4)}")
    # Forward + gradient sanity
    x = torch.randn(4, 64); y = torch.randint(0, 4, (4,))
    co.train()
    loss = torch.nn.functional.cross_entropy(co(x), y); loss.backward()
    nz = sum(1 for p in co.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    tot = sum(1 for p in co.parameters() if p.requires_grad)
    print(f"\ngradient sanity: {nz}/{tot} params received nonzero gradient, loss={loss.item():.4f}")
