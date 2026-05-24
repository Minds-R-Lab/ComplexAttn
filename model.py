"""
Resonant Manifold Cell (RMC) — proposed alternative to the affine-then-nonlinearity
primitive of neural networks.

Encoder produces (x0, p0) — initial position and momentum on a d-dim manifold.
The cell integrates Hamiltonian dynamics under
    H(x,p) = (1/2) p^T M^{-1} p + V(x),  V(x) = (1/2) x^T B x + V_MLP(x)
using leapfrog (Stormer-Verlet) for T steps. Output features are magnitudes of
windowed Fourier coefficients of the trajectory projected onto K mode directions
psi_k at frequencies omega_k. Selectivity comes from resonance, not a pointwise
nonlinearity. The integrator is symplectic so phase-space volume is conserved.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ResonantManifoldCell(nn.Module):

    def __init__(
        self,
        manifold_dim: int = 16,
        num_modes: int = 32,
        num_steps: int = 16,
        dt: float = 0.1,
        potential_hidden: int = 32,
        mlp_potential_scale: float = 0.3,
        eps_mass: float = 1e-3,
    ) -> None:
        super().__init__()
        self.d = manifold_dim
        self.K = num_modes
        self.T = num_steps
        self.dt = dt
        self.mlp_potential_scale = mlp_potential_scale

        # Inverse mass matrix M^{-1} = L L^T + eps I via lower-triangular L.
        L_init = torch.eye(self.d) * 0.5 + 0.02 * torch.randn(self.d, self.d).tril()
        self.L_raw = nn.Parameter(L_init)
        self.register_buffer("eps_I", eps_mass * torch.eye(self.d))

        # Quadratic potential matrix B (symmetric on use).
        self.B_raw = nn.Parameter(0.1 * torch.randn(self.d, self.d))

        # MLP correction to the potential.
        self.V_mlp = nn.Sequential(
            nn.Linear(self.d, potential_hidden),
            nn.Tanh(),
            nn.Linear(potential_hidden, 1, bias=False),
        )
        with torch.no_grad():
            self.V_mlp[-1].weight.mul_(0.05)

        # Resonant modes.
        self.psi = nn.Parameter(torch.randn(self.K, self.d) / math.sqrt(self.d))
        max_omega = math.pi / max(self.dt, 1e-6)
        self.omega = nn.Parameter(torch.linspace(0.2, 0.8 * max_omega, self.K))

    def mass_inv(self) -> torch.Tensor:
        L_tri = torch.tril(self.L_raw)
        return L_tri @ L_tri.T + self.eps_I

    def B_sym(self) -> torch.Tensor:
        return 0.5 * (self.B_raw + self.B_raw.T)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        B = self.B_sym()
        Bx = x @ B
        quad = 0.5 * (x * Bx).sum(dim=-1)
        mlp = self.mlp_potential_scale * self.V_mlp(x).squeeze(-1)
        return quad + mlp

    def grad_potential(self, x: torch.Tensor) -> torch.Tensor:
        # quadratic gradient is analytical
        B = self.B_sym()
        quad_grad = x @ B
        outer_grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            if not outer_grad_enabled:
                x_use = x.detach().requires_grad_(True)
                mlp_val = self.mlp_potential_scale * self.V_mlp(x_use).sum()
                (mlp_grad,) = torch.autograd.grad(mlp_val, x_use, create_graph=False)
            else:
                x_use = x if x.requires_grad else x.requires_grad_(True)
                mlp_val = self.mlp_potential_scale * self.V_mlp(x_use).sum()
                (mlp_grad,) = torch.autograd.grad(
                    mlp_val, x_use, create_graph=self.training, retain_graph=True
                )
        return quad_grad + mlp_grad

    def hamiltonian(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        M_inv = self.mass_inv()
        Mp = p @ M_inv
        kinetic = 0.5 * (p * Mp).sum(dim=-1)
        return kinetic + self.potential(x)

    def forward(self, x0: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
        M_inv = self.mass_inv()
        x, p = x0, p0
        traj = [x]
        for _ in range(self.T):
            gV = self.grad_potential(x)
            p_half = p - 0.5 * self.dt * gV
            x = x + self.dt * (p_half @ M_inv)
            gV_new = self.grad_potential(x)
            p = p_half - 0.5 * self.dt * gV_new
            traj.append(x)

        trajectory = torch.stack(traj, dim=0)
        n_steps = trajectory.shape[0]
        times = torch.arange(n_steps, device=x.device, dtype=x.dtype) * self.dt

        projections = trajectory @ self.psi.T
        phase_arg = self.omega.unsqueeze(0) * times.unsqueeze(1)
        cos_phase = torch.cos(phase_arg).unsqueeze(1)
        sin_phase = torch.sin(phase_arg).unsqueeze(1)
        a_cos = (projections * cos_phase).sum(dim=0) / n_steps
        a_sin = (projections * sin_phase).sum(dim=0) / n_steps
        return torch.sqrt(a_cos.pow(2) + a_sin.pow(2) + 1e-8)

    def integrate_with_trajectory(self, x0, p0):
        M_inv = self.mass_inv()
        x, p = x0, p0
        xs, ps = [x], [p]
        for _ in range(self.T):
            gV = self.grad_potential(x)
            p_half = p - 0.5 * self.dt * gV
            x = x + self.dt * (p_half @ M_inv)
            gV_new = self.grad_potential(x)
            p = p_half - 0.5 * self.dt * gV_new
            xs.append(x)
            ps.append(p)
        return torch.stack(xs, dim=0), torch.stack(ps, dim=0)


class RMCClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        manifold_dim: int = 16,
        num_modes: int = 32,
        num_classes: int = 10,
        num_steps: int = 16,
        dt: float = 0.1,
        potential_hidden: int = 32,
        momentum_init_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, 2 * manifold_dim)
        with torch.no_grad():
            self.encoder.weight[manifold_dim:].mul_(momentum_init_scale)
            self.encoder.bias[manifold_dim:].mul_(momentum_init_scale)
        self.rmc = ResonantManifoldCell(
            manifold_dim=manifold_dim,
            num_modes=num_modes,
            num_steps=num_steps,
            dt=dt,
            potential_hidden=potential_hidden,
        )
        self.head = nn.Linear(num_modes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        enc = self.encoder(x)
        x0, p0 = enc.chunk(2, dim=-1)
        features = self.rmc(x0, p0)
        return self.head(features)


class LinearBaseline(nn.Module):
    def __init__(self, input_dim: int = 784, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
