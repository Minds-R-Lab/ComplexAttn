"""Inspect the kernel near the expected peak location."""
import sys, os
sys.path.insert(0, "/home/claude/phase_negation/cdiag")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import (RealDiagSSM, ComplexDiagSSM,
                    _diag_ssm_kernel_real, _diag_ssm_kernel_complex)
from run import train


def get_kernel(model, L):
    with torch.no_grad():
        if isinstance(model, RealDiagSSM):
            return _diag_ssm_kernel_real(model.lam, model.B, model.C, L).cpu().numpy()
        return _diag_ssm_kernel_complex(model.lam, model.B, model.C, L).cpu().numpy()


device = "cuda" if torch.cuda.is_available() else "cpu"
K, L, A = 5, 150, 8
T = 2 * K + L + 1
steps = 1500 if device == "cpu" else 3000

models = {}
for kind, n in [("complex", 16), ("real", 128)]:
    torch.manual_seed(0); np.random.seed(0)
    if kind == "real":
        m = RealDiagSSM(n, A+2, A+1).to(device)
    else:
        m = ComplexDiagSSM(n, A+2, A+1).to(device)
    train(m, K, L, A, lr=5e-3, steps=steps, batch_size=64,
          device=device, log_every=steps, val_every=steps, tag=f"{kind}-n{n}")
    models[f"{kind}-n{n}"] = m

# Now: for each output class o in [1..8] (data classes) and input
# class i in [1..8], find the max position of |k[o, i, t]| and plot
# the diagonal o=i kernels (which are the "useful" ones).

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ideal_peak = K + L + 1   # 156

for ax, (tag, model) in zip(axes, models.items()):
    k = get_kernel(model, T)        # [d_out, d_in, T]
    # Plot all diagonal entries k[i, i, t] for i in 1..8 on the same axes
    for i in range(1, A + 1):
        kt = k[i, i, :]
        ax.plot(np.arange(T), kt, alpha=0.7, lw=1.0, label=f"k[{i},{i},:]" if i <= 3 else None)
    ax.axvline(ideal_peak, color="g", lw=1.2, ls="--",
               label=f"ideal delta at t={ideal_peak}", alpha=0.6)
    # Also mark the GO position
    ax.axvline(K + L, color="m", lw=1, ls=":",
               label=f"GO at t={K+L}", alpha=0.5)
    ax.set_ylabel(f"k[i, i, t]\n({tag})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
axes[-1].set_xlabel("Time lag t (steps)")
fig.suptitle("Diagonal kernel entries k[i, i, t] for data classes i=1..8")
plt.tight_layout()
plt.savefig("results_cdiag/diagonal_kernels.png", dpi=140)
print("Saved diagonal_kernels.png")

# Now look at GO -> output class kernels. The way the model probably
# implements copy is: when GO arrives, look back exactly L+1 steps for
# the data symbols. So k[o, GO_class, t] should have peaks at certain
# positions corresponding to the delays from data inputs to outputs.
# GO is at input position K+L = 155. Output positions are 156, 157, 158,
# 159, 160 = 155+1, 155+2, ..., 155+5. So GO -> output kernel should
# have peaks at t = 1, 2, 3, 4, 5.
# Actually NO: k[o, i, t] is the response of output channel o to input
# channel i at time lag t. If output at position 156 is set by GO at
# position 155 plus t=1, then k[*, GO, 1] should be the contribution.
# But the model would also need to remember which DATA token came before.
# Let me just plot k[o, GO_class, :] to see what's going on.
GO_class = A + 1                # = 9
fig2, ax = plt.subplots(figsize=(13, 4))
for tag, model in models.items():
    k = get_kernel(model, T)
    # Average |k[o, GO, t]| over output classes 1..8 (data classes)
    avg = np.abs(k[1:A+1, GO_class, :]).mean(axis=0)
    ax.plot(np.arange(T), avg, lw=1.5,
            label=f"{tag}: mean_o |k[o, GO, t]|", alpha=0.8)
ax.set_xlabel("Time lag t (steps)")
ax.set_ylabel("Mean |k[*, GO, t]| across output classes")
ax.set_title("Kernel from GO signal to output (averaged over output classes)")
ax.grid(alpha=0.3); ax.legend()
ax.set_xlim(0, 30)               # GO->output should be at small lags
plt.tight_layout()
plt.savefig("results_cdiag/go_kernel.png", dpi=140)
print("Saved go_kernel.png")

# Print the location of the max of |k[i, i, :]| for each data class
print("\nLocation of largest |k[i,i,t]| in [0..T] for each data class:")
for tag, model in models.items():
    k = get_kernel(model, T)
    print(f"\n{tag}:")
    for i in range(1, A + 1):
        kt = k[i, i, :]
        peak_idx = int(np.argmax(np.abs(kt)))
        peak_val = float(kt[peak_idx])
        print(f"  k[{i},{i},:]:  peak at t={peak_idx:3d}, value={peak_val:+.3f}")
