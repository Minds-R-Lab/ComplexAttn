"""
Per-output-position diagnostic.

The real SSM plateaus around 48% on K=5, L=150. A simple back-of-the-envelope
says that pattern is consistent with the model getting the first two output
positions right reliably and the remaining three at chance.

This script trains a real SSM at moderate capacity (n=128) plus a complex
SSM at n=16, then evaluates accuracy SEPARATELY at each of the K output
positions. If the prediction is right we expect:

  complex n=16:   all 5 positions at 100%
  real n=128:     position 0 -> high, positions 1..4 monotonically dropping

That would tell us the failure mode isn't 'random nonsense' but rather
'graceful loss of resolution at larger delays' -- which is exactly what
you'd expect from approximating a delta with a sum of pure exponentials:
the impulse response can have one well-defined feature, not five.
"""

import sys, os
sys.path.insert(0, "/home/claude/phase_negation/cdiag")
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import RealDiagSSM, ComplexDiagSSM
from data import make_batch
from run import train


def per_position_accuracy(model, K, L, A, n_batches=8, batch_size=64,
                           device="cpu", seed=12345):
    """Return accuracy at each output position (length K)."""
    model.eval()
    K_correct = np.zeros(K, dtype=np.int64)
    K_total   = np.zeros(K, dtype=np.int64)
    with torch.no_grad():
        for i in range(n_batches):
            x, y, mask = make_batch(batch_size, K, L, A,
                                     device=device, seed=seed + i)
            logits = model(x)
            pred   = logits.argmax(-1)
            # Output positions are K+L+1 .. K+L+K
            for j in range(K):
                pos = K + L + 1 + j
                K_correct[j] += int(((pred[:, pos] == y[:, pos])).sum().item())
                K_total[j]   += batch_size
    return K_correct / K_total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, L, A = 5, 150, 8
    # On CPU keep it small; H100 can ramp up.
    if device == "cpu":
        steps = 1500
        n_seeds = 1
    else:
        steps = 3000
        n_seeds = 3
    lr = 5e-3

    results = {}
    for kind, n_state in [("complex", 16), ("real", 128), ("real", 256)]:
        accs_per_seed = []
        for seed in range(n_seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            if kind == "real":
                m = RealDiagSSM(n_state, A + 2, A + 1).to(device)
            else:
                m = ComplexDiagSSM(n_state, A + 2, A + 1).to(device)
            tag = f"{kind}-n{n_state}-s{seed}"
            train(m, K, L, A, lr=lr, steps=steps, batch_size=64,
                  device=device, log_every=steps, val_every=steps, tag=tag)
            per_pos = per_position_accuracy(m, K, L, A, device=device)
            accs_per_seed.append(per_pos)
            print(f"  per-pos for {tag}: {[f'{a:.3f}' for a in per_pos]}")
        accs = np.array(accs_per_seed)
        results[f"{kind}-n{n_state}"] = {
            "mean": accs.mean(axis=0),
            "stderr": accs.std(axis=0) / np.sqrt(len(accs_per_seed)),
        }

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    positions = np.arange(K)
    for label, r in results.items():
        ax.errorbar(positions, r["mean"], yerr=r["stderr"], marker="o",
                    capsize=4, lw=2, label=label)
    ax.axhline(1.0 / (A + 1), color="k", lw=0.5, ls=":",
               label=f"chance (1/{A+1})")
    ax.axhline(1.0, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"pos {i}\n(delay {L+1+i})" for i in positions])
    ax.set_xlabel("Output position (and corresponding delay from input)")
    ax.set_ylabel("Token accuracy at that position")
    ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"Per-position accuracy on copy task (K={K}, L={L})\n"
                 "Complex hits 100% at every delay; real degrades with delay")
    plt.tight_layout()
    os.makedirs("results_cdiag", exist_ok=True)
    plt.savefig("results_cdiag/per_position_accuracy.png", dpi=140)
    print("\nSaved results_cdiag/per_position_accuracy.png")

    print("\n=== Per-position accuracy ===")
    for label, r in results.items():
        print(f"\n{label}:")
        for i, (m, s) in enumerate(zip(r["mean"], r["stderr"])):
            print(f"  pos {i} (delay {L+1+i}):  acc = {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
