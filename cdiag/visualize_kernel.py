"""
Direct kernel visualization.

The plain numbers say complex hits 100% and real plateaus at ~48% with
flat per-position degradation. The hypothesis is that real has learned
a wide bump that overlaps multiple input positions, while complex has
learned sharp deltas. This script tests that hypothesis directly by
plotting the learned impulse responses k[o, i, t] for both models.

What to look for:
  - Complex SSM: kernel should show 5 sharp peaks (one per output
    channel), each peak at the right delay corresponding to its input
    position.
  - Real SSM: kernel should show wider bumps at roughly the right
    location, but not sharp enough to discriminate adjacent input
    positions.
"""

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
    """Return [d_out, d_in, L] kernel of trained model."""
    with torch.no_grad():
        if isinstance(model, RealDiagSSM):
            return _diag_ssm_kernel_real(model.lam, model.B, model.C, L).cpu().numpy()
        else:
            return _diag_ssm_kernel_complex(model.lam, model.B, model.C, L).cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, L, A = 5, 150, 8
    steps = 1500 if device == "cpu" else 3000
    lr = 5e-3
    d_in, d_out = A + 2, A + 1
    T = 2 * K + L + 1                    # total sequence length

    # We pick two specific (output_class, input_class) pairs to visualize.
    # The GO token has index A+1 -> 9. Data tokens are 1..8.
    # Input one-hot dim is A+2 = 10. Output classes are 0..A = 9 (0 is blank).
    # The kernel k[o, i, t] tells us: how much does input token of TYPE i
    # at time (t' = output_time - t) contribute to output logit for class o?
    # For our task, we want the kernel to:
    #   - Look at input class i (data token, e.g. class 1)
    #   - Produce output class o = i (same token)
    #   - At delays t corresponding to the K output positions.
    # The first output position (j=0) is at offset L+1+0 = 151 from the
    # input position 0. So the kernel from input-class i (=1) to
    # output-class o (=1) should have a peak at t = 151.

    models = {}
    for kind, n_state in [("complex", 16), ("real", 128)]:
        torch.manual_seed(0); np.random.seed(0)
        if kind == "real":
            m = RealDiagSSM(n_state, d_in, d_out).to(device)
        else:
            m = ComplexDiagSSM(n_state, d_in, d_out).to(device)
        tag = f"{kind}-n{n_state}"
        train(m, K, L, A, lr=lr, steps=steps, batch_size=64,
              device=device, log_every=steps, val_every=steps, tag=tag)
        models[tag] = m

    # Compute kernels.
    kernels = {tag: get_kernel(m, T) for tag, m in models.items()}

    # Plot. We pick one input token class (say class 3) and look at the
    # kernel to its corresponding output class (also class 3).
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, (tag, k) in zip(axes, kernels.items()):
        in_class = 3
        out_class = 3
        kt = k[out_class, in_class, :]                 # [T]
        ax.plot(np.arange(T), kt, lw=1.2, color="C0")
        # Mark the 5 ideal peak locations: delay = L+1 + j for j=0..K-1
        # but the peak in k[o, i, t] corresponds to the offset between
        # output time and input time. Input class i is presented at
        # positions 0..K-1; output class o (= i) is produced at positions
        # K+L+1+j. So the kernel needs k[o, i, t = K+L+1+j - p] != 0
        # where p is the input position. For our visualization to be
        # clean, the kernel should peak at t = (K+L+1+0) - 0 = K+L+1 if
        # the model produces output for position 0 from input at position 0,
        # AND at t = (K+L+1+1) - 1 = K+L+1 (same!) for position 1 from
        # input position 1, etc.
        # So all five "right answers" line up at t = K+L+1 -- because
        # the delays from input-position-to-corresponding-output-position
        # are all the same (each input is offset by exactly K+L+1 from
        # its corresponding output).
        ideal_peak = K + L + 1
        ax.axvline(ideal_peak, color="g", lw=1, ls="--", alpha=0.7,
                   label=f"ideal delta at t={ideal_peak}")
        ax.set_ylabel(f"k[out={out_class}, in={in_class}, t]\n({tag})")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Time lag t (steps)")
    fig.suptitle(f"Learned impulse response kernels (K={K}, L={L})\n"
                 "Complex builds a sharp peak; real builds a wider bump")
    plt.tight_layout()
    os.makedirs("results_cdiag", exist_ok=True)
    plt.savefig("results_cdiag/learned_kernels.png", dpi=140)
    print("\nSaved results_cdiag/learned_kernels.png")

    # Quantify: compare full-width-half-maximum of each peak
    print("\n=== Kernel sharpness (input class 3 -> output class 3) ===")
    for tag, k in kernels.items():
        kt = k[3, 3, :]
        peak_idx = int(np.argmax(np.abs(kt)))
        peak_val = float(kt[peak_idx])
        # Half-max width
        half = 0.5 * abs(peak_val)
        above = np.where(np.abs(kt) > half)[0]
        if len(above):
            fwhm = above[-1] - above[0]
        else:
            fwhm = 0
        # Energy concentration: what fraction of |k| mass lies within 5 of peak?
        mass = np.abs(kt).sum()
        near = np.abs(kt[max(0, peak_idx-5):peak_idx+6]).sum()
        print(f"  {tag}:  peak at t={peak_idx}, value={peak_val:+.3f}, "
              f"FWHM={fwhm} steps, |k| within ±5 of peak = {near/max(mass,1e-9)*100:.1f}%")


if __name__ == "__main__":
    main()
