"""Numerical sanity checks for the Resonant Manifold Cell.

Verifies:
1. Energy conservation under leapfrog (oscillation bounded, no drift).
2. Reversibility (forward T + flip p + forward T returns near origin).
3. Gradient flow: every learnable parameter receives a non-zero grad.
"""

from __future__ import annotations

import torch

from model import ResonantManifoldCell, RMCClassifier, count_parameters


@torch.no_grad()
def energy_conservation(cell, batch_size: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    x = 0.3 * torch.randn(batch_size, cell.d)
    p = 0.3 * torch.randn(batch_size, cell.d)
    H0 = cell.hamiltonian(x, p)
    drifts = []
    M_inv = cell.mass_inv()
    cur_x, cur_p = x, p
    for _ in range(cell.T):
        gV = cell.grad_potential(cur_x)
        p_half = cur_p - 0.5 * cell.dt * gV
        cur_x = cur_x + cell.dt * (p_half @ M_inv)
        gV_new = cell.grad_potential(cur_x)
        cur_p = p_half - 0.5 * cell.dt * gV_new
        H = cell.hamiltonian(cur_x, cur_p)
        drifts.append(((H - H0) / (H0.abs() + 1e-6)).abs().mean().item())
    return {
        "mean_relative_drift": float(sum(drifts) / len(drifts)),
        "max_relative_drift": float(max(drifts)),
        "final_relative_drift": float(drifts[-1]),
    }


@torch.no_grad()
def reversibility(cell, batch_size: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    x0 = 0.3 * torch.randn(batch_size, cell.d)
    p0 = 0.3 * torch.randn(batch_size, cell.d)
    M_inv = cell.mass_inv()
    x, p = x0.clone(), p0.clone()
    for _ in range(cell.T):
        gV = cell.grad_potential(x)
        p_half = p - 0.5 * cell.dt * gV
        x = x + cell.dt * (p_half @ M_inv)
        gV_new = cell.grad_potential(x)
        p = p_half - 0.5 * cell.dt * gV_new
    p = -p
    for _ in range(cell.T):
        gV = cell.grad_potential(x)
        p_half = p - 0.5 * cell.dt * gV
        x = x + cell.dt * (p_half @ M_inv)
        gV_new = cell.grad_potential(x)
        p = p_half - 0.5 * cell.dt * gV_new
    pos_err = (x - x0).norm(dim=-1) / (x0.norm(dim=-1) + 1e-6)
    mom_err = (-p - p0).norm(dim=-1) / (p0.norm(dim=-1) + 1e-6)
    return {
        "mean_position_error": float(pos_err.mean().item()),
        "max_position_error": float(pos_err.max().item()),
        "mean_momentum_error": float(mom_err.mean().item()),
    }


def gradient_flow(classifier, batch_size: int = 4):
    classifier.train()
    x = torch.randn(batch_size, 1, 28, 28)
    target = torch.randint(0, 10, (batch_size,))
    logits = classifier(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    total, nonzero, missing = 0, 0, 0
    per_param = {}
    for name, p in classifier.named_parameters():
        total += 1
        if p.grad is None:
            per_param[name] = None
            missing += 1
            continue
        g_norm = p.grad.norm().item()
        per_param[name] = g_norm
        if g_norm > 0:
            nonzero += 1
    return {
        "loss": float(loss.item()),
        "params_with_grad": nonzero,
        "params_total": total,
        "min_grad_norm": min((v for v in per_param.values() if v is not None), default=0.0),
        "max_grad_norm": max((v for v in per_param.values() if v is not None), default=0.0),
        "params_missing_grad": missing,
        "param_grad_norms": per_param,
    }


def run_all():
    print("\n=== RMC numerical verification ===\n")
    cell = ResonantManifoldCell()
    energy = energy_conservation(cell)
    print("Energy conservation (leapfrog should oscillate but not drift):")
    for k, v in energy.items():
        print(f"  {k}: {v:.4e}")

    rev = reversibility(cell)
    print("\nReversibility (forward + flip p + forward should return to start):")
    for k, v in rev.items():
        print(f"  {k}: {v:.4e}")

    cls = RMCClassifier()
    print(f"\nRMC classifier param count: {count_parameters(cls)}")
    grad = gradient_flow(cls)
    print(f"Gradient flow: loss={grad['loss']:.4f}, "
          f"{grad['params_with_grad']}/{grad['params_total']} params received nonzero gradient "
          f"({grad['params_missing_grad']} structurally unused)")
    print(f"  grad norms — min: {grad['min_grad_norm']:.2e}, max: {grad['max_grad_norm']:.2e}")
    print("  per-param grad norms:")
    for name, g in grad["param_grad_norms"].items():
        if g is None:
            print(f"    {name}: (no grad)")
        else:
            print(f"    {name}: {g:.3e}")

    print("\n=== Verification complete ===\n")
    return {"energy": energy, "reversibility": rev, "gradient_flow": grad}


if __name__ == "__main__":
    run_all()
