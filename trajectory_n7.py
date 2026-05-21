"""
Trajectory analysis on the unexplained 2-layer GRU partial escape at n=7.

Experiment 5 found that 2-layer GRU at d=128 on Z/7 hits OOD 0.79
(when single-layer GRU and all LSTM at the same scale stay at chance).
The mechanism we identified for the Z/5 GRU failure was state
saturation against the tanh ceiling. Two questions:

  Q1. Does the 2-layer GRU on n=7 show the same saturation pattern,
      or a different one? If it saturates but still hits 0.79 OOD,
      the second layer is doing something that the readout
      apparently makes use of. If it does NOT saturate, the
      mechanism story has to be expanded.

  Q2. Per-k, does argmax invariance show the same U-shape as the
      1-layer Z/5 GRU (high near training, zero in transition,
      recovering at saturation)? Or a different shape?

We use the same diagnostics as trajectory_analysis.py:
  - state magnitude vs k
  - state ||Delta|| under 7*TWIRL (per fixed base sequence)
  - logit ||Delta|| under 7*TWIRL
  - argmax invariance per k

Comparison architecture: PhaseSumNet at d=16 on n=7, which we
already know hits 1.000 OOD. Useful as the reference for what
"truly internalized Z/7" looks like in the diagnostics.

For extracting per-step state from the 2-layer GRU we have to be a
little careful: nn.GRU returns the output of the TOP layer at every
step (which is what we want for the readout) and the final hidden
state at every layer (which is what the readout actually uses).
"""

import math
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_cyclic import CyclicTaskSpec, CLS_ID, PAD_ID, NUM_FILLERS
from models_capacity import GRUMultilayer, PhaseSumRef
from train_cyclic import train_model


OUTPUT_DIR = "results_trajectory_n7"


# ---------- probe utilities (parallel to trajectory_analysis.py) ----------

def make_probe_sequences(spec, k_values, n_samples=64, base_len=8, seed=42):
    g = torch.Generator().manual_seed(seed)
    bases = []
    for _ in range(n_samples):
        atom = spec.atom_ids[int(torch.randint(0, spec.n, (1,),
                                                 generator=g).item())]
        seq = [CLS_ID, atom]
        for _ in range(base_len - 2):
            seq.append(int(spec.filler_start +
                            torch.randint(0, NUM_FILLERS, (1,),
                                          generator=g).item()))
        bases.append(seq)
    L_max = base_len + max(k_values)
    by_k = {}
    for k in k_values:
        toks = torch.full((n_samples, L_max), PAD_ID, dtype=torch.long)
        for i, base in enumerate(bases):
            seq = base + [spec.twirl_id] * k
            toks[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        by_k[k] = toks
    return by_k


def gru_top_layer_state(model, tokens):
    """For a (possibly multi-layer) bidirectional GRU, return the
    TOP-layer forward hidden state at the last non-PAD token.

    The readout (model.head) sees `torch.cat([h_n[-2], h_n[-1]], -1)`
    where h_n[-2] is the top-layer forward, h_n[-1] is the top-layer
    backward. We probe the forward half, since that's what feels the
    cumulative TWIRL effect.
    """
    with torch.no_grad():
        x = model.embed(tokens)
        full_output, h_n = model.gru(x)               # full_output: [B,L,2d]
        # full_output's last dim splits as [forward_out | backward_out]
        d = model.d_model
        fwd_out = full_output[:, :, :d]
        non_pad = (tokens != PAD_ID).long()
        last_idx = non_pad.sum(dim=1) - 1
        idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, d)
        return fwd_out.gather(1, idx).squeeze(1)


def phasesum_state(model, tokens):
    with torch.no_grad():
        phases = model.phase(tokens)
        non_pad = (tokens != PAD_ID).float().unsqueeze(-1)
        phases = phases * non_pad
        cum = phases.cumsum(dim=1)
        last_idx = (tokens != PAD_ID).long().sum(dim=1) - 1
        idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, cum.size(-1))
        total = cum.gather(1, idx).squeeze(1)
        return torch.stack([torch.cos(total),
                             torch.sin(total)], dim=-1).reshape(total.size(0), -1)


def collect(model, tokens_by_k, kind):
    out = {}
    for k, tokens in tokens_by_k.items():
        h = (gru_top_layer_state(model, tokens) if kind == "gru"
              else phasesum_state(model, tokens))
        out[k] = h.cpu().numpy()
    return out


def magnitude(states):
    ks = sorted(states)
    means = [np.linalg.norm(states[k], axis=-1).mean() for k in ks]
    ses   = [np.linalg.norm(states[k], axis=-1).std() / math.sqrt(states[k].shape[0])
              for k in ks]
    return np.array(ks), np.array(means), np.array(ses)


def state_return(states, period):
    ks_avail = sorted(states)
    ks, abs_, rel_ = [], [], []
    for k in ks_avail:
        if k + period not in states:
            continue
        diffs = states[k + period] - states[k]
        norms = np.linalg.norm(diffs, axis=-1)
        ref = 0.5 * (np.linalg.norm(states[k], axis=-1).mean() +
                      np.linalg.norm(states[k + period], axis=-1).mean())
        ks.append(k); abs_.append(float(norms.mean()))
        rel_.append(float(norms.mean() / max(ref, 1e-8)))
    return np.array(ks), np.array(abs_), np.array(rel_)


def logit_return(model, tokens_by_k, period, device):
    cache = {}
    model.eval()
    with torch.no_grad():
        for k, t in tokens_by_k.items():
            cache[k] = model(t).cpu().numpy()
    ks, l2, inv = [], [], []
    for k in sorted(cache):
        if k + period not in cache:
            continue
        diff = cache[k + period] - cache[k]
        l2.append(float(np.linalg.norm(diff, axis=-1).mean()))
        pred_k  = cache[k].argmax(-1)
        pred_kp = cache[k + period].argmax(-1)
        inv.append(float((pred_k == pred_kp).mean()))
        ks.append(k)
    return np.array(ks), np.array(l2), np.array(inv)


# ---------- plots ----------

def plot_argmax_invariance_comparison(curves, path):
    """Three curves on one plot: 2-layer GRU on n=7 (the escape),
       PhaseSumNet on n=7 (the reference), and (for context) the
       1-layer GRU result on n=5 loaded from results_trajectory."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, color, ks_inv in curves:
        ks, inv = ks_inv
        ax.plot(ks, inv, marker="o", color=color, lw=2, label=label)
    ax.axhline(1.0 / 7, color="gray", lw=0.6, ls=":", label="Chance (1/7) for Z/7")
    ax.axhline(1.0 / 5, color="lightgray", lw=0.6, ls=":", label="Chance (1/5) for Z/5")
    ax.set_xlabel("k")
    ax.set_ylabel("Fraction of sequences with argmax(k+n)=argmax(k)")
    ax.set_title("Argmax invariance comparison: where the 2-layer GRU sits between failure and success")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_state_magnitude(ks_gru, mag_gru, mag_se_gru,
                          ks_ps,  mag_ps,  mag_se_ps,
                          path):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.fill_between(ks_gru, mag_gru - mag_se_gru, mag_gru + mag_se_gru,
                    color="C2", alpha=0.18)
    ax.plot(ks_gru, mag_gru, marker="o", color="C2", lw=2,
            label="2-layer GRU d=128 on Z/7 (escape model)")
    ax.fill_between(ks_ps, mag_ps - mag_se_ps, mag_ps + mag_se_ps,
                    color="C0", alpha=0.18)
    ax.plot(ks_ps, mag_ps, marker="s", color="C0", lw=2,
            label="PhaseSumNet d=16 on Z/7")
    ax.axvspan(0, 5, color="gray", alpha=0.10, label="Train distribution (k≤5)")
    ax.set_xlabel("k"); ax.set_ylabel("Mean state magnitude")
    ax.set_title("State magnitude — does the 2-layer GRU saturate?")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_logit_closure(curves, period, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for label, color, (ks, l2, inv) in curves:
        axes[0].plot(ks, l2, marker="o", color=color, lw=2, label=label)
        axes[1].plot(ks, inv, marker="o", color=color, lw=2, label=label)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel(f"Mean ||logits(k+{period}) − logits(k)||")
    axes[0].set_title("Logit closure (absolute)")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)
    axes[1].axhline(1.0, color="k", lw=0.5, ls=":", label="Perfect invariance")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel(f"argmax(k+{period}) == argmax(k)")
    axes[1].set_title(f"Argmax invariance under {period}·TWIRL")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
    fig.suptitle(f"Z/7 trajectory diagnostics: 2-layer GRU vs PhaseSumNet")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


# ---------- main ----------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    spec = CyclicTaskSpec(7)

    print("\n>>> Training 2-layer GRU (d=128, L=2, Z/7)  -- the escape cell")
    gru = GRUMultilayer(spec, d_model=128, n_layers=2)
    gru, _ = train_model(
        gru, spec,
        train_depth_range=(0, 5),
        eval_depths=tuple(range(0, 21)),
        n_train=120_000, n_eval_per_depth=2_000,
        batch_size=256, lr=3e-3, n_epochs=25,
        eval_every_steps=500, device=device, seed=0,
        tag="GRU-L2-d128-n7")

    print("\n>>> Training PhaseSumNet (d=16, Z/7) -- the reference")
    ps = PhaseSumRef(spec, d_model=16)
    ps, _ = train_model(
        ps, spec,
        train_depth_range=(0, 5),
        eval_depths=tuple(range(0, 21)),
        n_train=60_000, n_eval_per_depth=2_000,
        batch_size=256, lr=3e-3, n_epochs=15,
        eval_every_steps=500, device=device, seed=0,
        tag="PhaseSum-n7")

    gru.eval(); ps.eval()

    k_values = list(range(0, 31))
    tokens_by_k = make_probe_sequences(spec, k_values, n_samples=64)
    tokens_by_k = {k: t.to(device) for k, t in tokens_by_k.items()}

    gru_states = collect(gru, tokens_by_k, "gru")
    ps_states  = collect(ps,  tokens_by_k, "phasesum")

    ks_g, mag_g, mag_se_g = magnitude(gru_states)
    ks_p, mag_p, mag_se_p = magnitude(ps_states)

    plot_state_magnitude(ks_g, mag_g, mag_se_g, ks_p, mag_p, mag_se_p,
                          os.path.join(OUTPUT_DIR, "magnitude.png"))

    gru_lr = logit_return(gru, tokens_by_k, 7, device)
    ps_lr  = logit_return(ps,  tokens_by_k, 7, device)
    plot_logit_closure(
        [("2-layer GRU d=128 (Z/7)", "C2", gru_lr),
          ("PhaseSumNet d=16 (Z/7)",  "C0", ps_lr)],
        7, os.path.join(OUTPUT_DIR, "logit_closure.png"))

    # Comparison plot pulling in the Z/5 GRU result if available.
    curves = [
        ("2-layer GRU d=128 on Z/7  (OOD ≈ 0.79)", "C2",
          (gru_lr[0], gru_lr[2])),
        ("PhaseSumNet d=16 on Z/7  (OOD = 1.00)",  "C0",
          (ps_lr[0],  ps_lr[2])),
    ]
    prior = "results_trajectory/summary.json"
    if os.path.exists(prior):
        with open(prior) as f:
            prior_summary = json.load(f)
        if "GRU" in prior_summary:
            inv_z5 = prior_summary["GRU"].get("argmax_invariance_5", {})
            if inv_z5:
                ks_5 = np.array(sorted(int(k) for k in inv_z5))
                vals = np.array([inv_z5[str(k)] for k in ks_5])
                curves.append(
                    ("1-layer GRU d=64 on Z/5  (OOD ≈ 0.25, the failing baseline)",
                     "C3", (ks_5, vals)))
    plot_argmax_invariance_comparison(curves,
        os.path.join(OUTPUT_DIR, "invariance_comparison.png"))

    # ---- numerical summary ----
    sg_abs, sg_rel = state_return(gru_states, 7)[1], state_return(gru_states, 7)[2]
    sp_abs, sp_rel = state_return(ps_states, 7)[1],  state_return(ps_states, 7)[2]
    summary = {
        "gru_L2_d128_n7": {
            "magnitude_at_k":        {int(k): float(m) for k, m in zip(ks_g, mag_g)},
            "state_return_7_abs":    [float(x) for x in sg_abs],
            "state_return_7_rel":    [float(x) for x in sg_rel],
            "logit_dist_7":          [float(x) for x in gru_lr[1]],
            "argmax_invariance_7":   {int(k): float(v) for k, v in
                                        zip(gru_lr[0], gru_lr[2])},
            "mean_state_magnitude_OOD":     float(mag_g[6:].mean()),
            "magnitude_ratio_30_over_5":    float(mag_g[ks_g.tolist().index(30)] /
                                                    mag_g[ks_g.tolist().index(5)]),
            "mean_state_return_7_rel":       float(sg_rel.mean()),
            "mean_logit_dist_7":             float(gru_lr[1].mean()),
            "mean_argmax_invariance_7":      float(gru_lr[2].mean()),
        },
        "phasesum_d16_n7": {
            "magnitude_at_k":        {int(k): float(m) for k, m in zip(ks_p, mag_p)},
            "state_return_7_abs":    [float(x) for x in sp_abs],
            "state_return_7_rel":    [float(x) for x in sp_rel],
            "logit_dist_7":          [float(x) for x in ps_lr[1]],
            "argmax_invariance_7":   {int(k): float(v) for k, v in
                                        zip(ps_lr[0], ps_lr[2])},
            "mean_state_magnitude_OOD":     float(mag_p[6:].mean()),
            "magnitude_ratio_30_over_5":    float(mag_p[ks_p.tolist().index(30)] /
                                                    mag_p[ks_p.tolist().index(5)]),
            "mean_state_return_7_rel":       float(sp_rel.mean()),
            "mean_logit_dist_7":             float(ps_lr[1].mean()),
            "mean_argmax_invariance_7":      float(ps_lr[2].mean()),
        },
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Z/7 trajectory diagnostics ===")
    for name, s in summary.items():
        print(f"\n{name}:")
        print(f"  mean state magnitude k>5:          {s['mean_state_magnitude_OOD']:.4f}")
        print(f"  magnitude ratio  30 / 5:           {s['magnitude_ratio_30_over_5']:.4f}")
        print(f"  state ||Δ|| / ||h||  (rel):        {s['mean_state_return_7_rel']:.4f}")
        print(f"  logit ||Δ||:                       {s['mean_logit_dist_7']:.4f}")
        print(f"  argmax invariance after 7·TWIRL:   {s['mean_argmax_invariance_7']:.4f}")


if __name__ == "__main__":
    main()
