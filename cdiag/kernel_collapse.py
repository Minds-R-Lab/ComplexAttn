"""Check whether the real model's kernels k[o, i, :] really are similar
across different (o, i) pairs, or whether they look class-specific.

Quantify by computing the matrix of pairwise correlations between all
diagonal kernels k[i, i, :] for i in [1..8].
"""
import sys, os
sys.path.insert(0, "/home/claude/phase_negation/cdiag")
import numpy as np
import torch
from models import (RealDiagSSM, ComplexDiagSSM, RealRoPESSM,
                    _diag_ssm_kernel_real, _diag_ssm_kernel_complex,
                    _diag_ssm_kernel_real_rope)
from run import train


def get_kernel(model, L):
    with torch.no_grad():
        if isinstance(model, RealDiagSSM):
            return _diag_ssm_kernel_real(model.lam, model.B, model.C, L).cpu().numpy()
        if isinstance(model, RealRoPESSM):
            return _diag_ssm_kernel_real_rope(
                model.decay, model.theta,
                model.B1, model.B2, model.C1, model.C2, L).cpu().numpy()
        return _diag_ssm_kernel_complex(model.lam, model.B, model.C, L).cpu().numpy()


device = "cuda" if torch.cuda.is_available() else "cpu"
K, L, A = 5, 150, 8
T = 2 * K + L + 1
steps = 1500 if device == "cpu" else 3000

results = {}
configs = [
    ("complex",          16),
    ("real",             128),
    ("real_rope",        16),
    ("real_rope_frozen", 16),
    ("real_rope_neg",    16),
]
for kind, n in configs:
    torch.manual_seed(0); np.random.seed(0)
    if kind == "real":
        m = RealDiagSSM(n, A+2, A+1).to(device)
    elif kind == "complex":
        m = ComplexDiagSSM(n, A+2, A+1).to(device)
    elif kind == "real_rope":
        m = RealRoPESSM(n, A+2, A+1, learn_theta=True).to(device)
    elif kind == "real_rope_frozen":
        m = RealRoPESSM(n, A+2, A+1, learn_theta=False).to(device)
    elif kind == "real_rope_neg":
        m = RealRoPESSM(n, A+2, A+1, learn_theta=True,
                         allow_negative_decay=True).to(device)
    train(m, K, L, A, lr=5e-3, steps=steps, batch_size=64,
          device=device, log_every=steps, val_every=steps, tag=f"{kind}-n{n}")
    k = get_kernel(m, T)
    diag = np.stack([k[i, i, :] for i in range(1, A+1)], axis=0)
    results[f"{kind}-n{n}"] = diag

print("\n=== Correlation matrix between diagonal kernels k[i,i,:] for i=1..8 ===")
for tag, diag in results.items():
    # Normalize each row
    norms = np.linalg.norm(diag, axis=1, keepdims=True)
    normed = diag / np.maximum(norms, 1e-9)
    corr = normed @ normed.T          # [8, 8]
    mean_offdiag = (corr.sum() - 8) / (8*8 - 8)
    print(f"\n{tag}:")
    print(f"  Mean off-diagonal correlation: {mean_offdiag:.4f}")
    print(f"  Min off-diagonal correlation: {(corr - np.eye(8)).min():.4f}")
    # Show the matrix briefly
    for i in range(8):
        print("  " + " ".join(f"{c:+.2f}" for c in corr[i]))

# Also: what is the "effective rank" of the kernel matrix [8, T]?
print("\n=== Effective rank (numerical rank of [8, T] matrix) ===")
for tag, diag in results.items():
    U, s, _ = np.linalg.svd(diag, full_matrices=False)
    # Effective rank via stable rank: sum(s^2) / max(s)^2
    stable_rank = (s**2).sum() / max(s[0]**2, 1e-9)
    # Or: count singular values within 1% of the top
    n_significant = int((s > 0.01 * s[0]).sum())
    print(f"  {tag}: top singular values = {s[:5]}, "
          f"stable_rank={stable_rank:.2f}, n_significant_1pct={n_significant}")
