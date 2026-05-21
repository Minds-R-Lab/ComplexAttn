"""
State-trajectory analysis: WHY does the GRU fail on Z/5?

Trains one GRU and one PhaseSumNet on Z/5 (the regime where Exp 5
showed the structural barrier), then probes their hidden-state
trajectories as a function of TWIRL count k. Three diagnostics:

  1. STATE MAGNITUDE ||h(k)|| vs k.  A model that has internalized
     a cyclic group should have a state on a bounded orbit.
     Divergence -> no limit cycle. Saturation -> bounded but maybe
     stuck.

  2. STATE TRAJECTORY in 2D PCA.  PhaseSumNet's state, restricted to
     cos(theta) and sin(theta) for a single phase dim, lives on the
     unit circle by construction. The GRU's tanh-bounded state has
     no such constraint. We expect: PhaseSumNet traces exactly 5
     equispaced points on the unit circle (Z/5 attractor); GRU
     traces something that does NOT close into a 5-cycle.

  3. MODULAR RETURN ||h(k+5) - h(k)||.  The defining property of an
     architecture that has Z/5 internalized is that adding 5 TWIRLs
     returns the state. For each k from 0..25, we measure how close
     h(k+5) is to h(k). Near zero means closure; large means failure.

The trained models from Exp 5 aren't saved, so we train fresh ones
here with the same hyperparameters. This is fast on CPU.
"""

import math
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_cyclic import (CyclicTaskSpec, CLS_ID, PAD_ID,
                          NUM_FILLERS)
from models_capacity import GRUMultilayer, PhaseSumRef
from train_cyclic import train_model


OUTPUT_DIR = "results_trajectory"


def make_probe_sequences(spec: CyclicTaskSpec, k_values, n_samples=64,
                          base_len=8, seed=42):
    """Build paired sequences differing only in number of TWIRL tokens.

    Each sequence has [CLS, atom, fillers...] of length base_len, plus
    k trailing TWIRLs.  Keeping the base prefix identical across k means
    any difference in hidden state at the final position is attributable
    to the TWIRL count and not to incidental sequence content.
    """
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

    by_k = {}
    L_max = base_len + max(k_values)
    for k in k_values:
        toks = torch.full((n_samples, L_max), PAD_ID, dtype=torch.long)
        for i, base in enumerate(bases):
            seq = base + [spec.twirl_id] * k
            toks[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        by_k[k] = toks
    return by_k


def gru_states_at_final_token(model, tokens):
    """Extract the forward GRU hidden state at the LAST non-PAD token,
       for each sequence in the batch.
    """
    with torch.no_grad():
        x = model.embed(tokens)
        # We run only the forward direction by hand so we can grab the
        # state at every step. nn.GRU is bidirectional; pull just the
        # forward weights.
        full_output, _ = model.gru(x)   # [B, L, 2d]
        d = model.d_model
        fwd_states = full_output[:, :, :d]            # forward direction
        # last non-PAD position per sequence
        non_pad = (tokens != PAD_ID).long()
        last_idx = non_pad.sum(dim=1) - 1             # [B]
        idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, d)
        last_h = fwd_states.gather(1, idx).squeeze(1)  # [B, d]
        return last_h


def phasesum_states_at_final_token(model, tokens):
    """For PhaseSumNet there's no per-step state; the relevant 'state'
       is the cumulative sum of phases up to (and including) the last
       non-PAD token. We compute it directly.
    """
    with torch.no_grad():
        phases = model.phase(tokens)                  # [B, L, d]
        # cumulative sum, mask PAD positions to zero
        non_pad = (tokens != PAD_ID).float().unsqueeze(-1)
        phases = phases * non_pad
        cum = phases.cumsum(dim=1)                    # [B, L, d]
        last_idx = (tokens != PAD_ID).long().sum(dim=1) - 1
        idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, cum.size(-1))
        total = cum.gather(1, idx).squeeze(1)          # [B, d]
        # Return cos/sin pair for visualization in 2D.
        return torch.stack([torch.cos(total),
                             torch.sin(total)], dim=-1).reshape(total.size(0), -1)


def collect_trajectories(model, tokens_by_k, kind):
    """Return dict k -> state array [n_samples, state_dim]."""
    states = {}
    for k, tokens in tokens_by_k.items():
        if kind == "gru":
            h = gru_states_at_final_token(model, tokens)
        elif kind == "phasesum":
            h = phasesum_states_at_final_token(model, tokens)
        else:
            raise ValueError(kind)
        states[k] = h.cpu().numpy()
    return states


def magnitude_curve(states):
    """k -> mean ||h(k)||_2 across samples, plus stderr."""
    ks = sorted(states)
    means, stderrs = [], []
    for k in ks:
        norms = np.linalg.norm(states[k], axis=-1)
        means.append(float(norms.mean()))
        stderrs.append(float(norms.std() / math.sqrt(len(norms))))
    return np.array(ks), np.array(means), np.array(stderrs)


def logit_modular_return(model, tokens_by_k, period, device):
    """How does adding `period` TWIRLs change the LOGITS?

    This is the truly readout-relevant question. State-level closure
    can be deceiving: PhaseSumNet's 16-dim phase state moves a lot
    even after closure because the linear readout only cares about
    cos/sin combinations. The logits are what the argmax sees.

    We return both absolute logit distance and the fraction of
    sequences whose argmax is invariant after adding `period`
    TWIRLs.
    """
    ks_avail = sorted(tokens_by_k)
    rows = []
    model.eval()
    cache = {}
    with torch.no_grad():
        for k in ks_avail:
            cache[k] = model(tokens_by_k[k]).cpu().numpy()
    for k in ks_avail:
        if k + period not in cache:
            continue
        diff = cache[k + period] - cache[k]
        l2 = np.linalg.norm(diff, axis=-1)
        pred_k  = cache[k].argmax(-1)
        pred_kp = cache[k + period].argmax(-1)
        argmax_invariant = (pred_k == pred_kp).mean()
        rows.append((k, float(l2.mean()), float(argmax_invariant)))
    ks   = np.array([r[0] for r in rows])
    l2   = np.array([r[1] for r in rows])
    inv  = np.array([r[2] for r in rows])
    return ks, l2, inv


def plot_logit_closure(gru_logit, ps_logit, period, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    for (ks, l2, inv), color, label in [
        (gru_logit, "C3", "GRU"),
        (ps_logit,  "C0", "PhaseSumNet"),
    ]:
        axes[0].plot(ks, l2, marker="o", color=color, lw=2, label=label)
        axes[1].plot(ks, inv, marker="o", color=color, lw=2, label=label)
    axes[0].set_xlabel("k"); axes[0].set_ylabel(f"Mean ||logits(k+{period}) − logits(k)||")
    axes[0].set_title("Logit closure (absolute)")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].axhline(1.0, color="k", lw=0.5, ls=":", label="Perfect invariance")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Fraction of sequences with argmax(k+5) = argmax(k)")
    axes[1].set_title("Argmax invariance under 5·TWIRL")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle(f"Logit-level closure under {period}·TWIRL — the diagnostic that actually predicts accuracy")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def modular_return_curve(states, period):
    """For each k where both k and k+period are present, mean ||h(k+p) - h(k)||.

    CRITICAL: states[k] and states[k+p] are arranged so that index i in
    both refers to the SAME base sequence (the probe builder keeps base
    prefixes identical across k). So the diff is per-sequence, not
    cross-sequence. This is the meaningful "did adding p TWIRLs leave
    me where I was" measurement.

    We also normalize by the mean state magnitude to give a relative
    closure score: 0 = perfect closure, 1 = displacement equal to a
    full state vector's worth.
    """
    ks_avail = sorted(states)
    ks, dists, stderrs, normed = [], [], [], []
    for k in ks_avail:
        if k + period not in states:
            continue
        diffs = states[k + period] - states[k]
        norms = np.linalg.norm(diffs, axis=-1)
        # Reference magnitude: mean over both k and k+p
        ref_mag = 0.5 * (np.linalg.norm(states[k],          axis=-1).mean() +
                          np.linalg.norm(states[k + period], axis=-1).mean())
        ks.append(k)
        dists.append(float(norms.mean()))
        stderrs.append(float(norms.std() / math.sqrt(len(norms))))
        normed.append(float(norms.mean() / max(ref_mag, 1e-8)))
    return np.array(ks), np.array(dists), np.array(stderrs), np.array(normed)


def project_pca(states, k_values):
    """Stack [k0, k1, ...] x [B, d] -> 2D PCA of all rows.

    Returns:
      proj    -> dict k -> [B, 2]
      explained_var -> two-vector of fraction of variance per component
    """
    all_rows = np.concatenate([states[k] for k in k_values], axis=0)
    mean = all_rows.mean(axis=0, keepdims=True)
    centered = all_rows - mean
    # numpy SVD: U @ diag(s) @ Vt = centered
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    pcs = Vt[:2]                                       # [2, d]
    var = (s ** 2) / (centered.shape[0] - 1)
    explained = var / var.sum()
    proj = {}
    n_each = states[k_values[0]].shape[0]
    for i, k in enumerate(k_values):
        block = centered[i * n_each : (i + 1) * n_each]
        proj[k] = block @ pcs.T                       # [B, 2]
    return proj, explained[:2]


def plot_magnitude(gru_curve, ps_curve, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (ks, ms, ss), color, label in [
        (gru_curve, "C3", "GRU (d=64, L=1)  — hidden state norm"),
        (ps_curve,  "C0", "PhaseSumNet (d=16)  — [cos,sin] norm"),
    ]:
        ax.fill_between(ks, ms - ss, ms + ss, color=color, alpha=0.20)
        ax.plot(ks, ms, marker="o", color=color, lw=2, label=label)
    ax.axvspan(0, 5, color="gray", alpha=0.10, label="Train distribution (k ≤ 5)")
    ax.set_xlabel("Number of TWIRL tokens (k)")
    ax.set_ylabel("Mean state magnitude")
    ax.set_title("State magnitude as a function of TWIRL count (Z/5)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=140); plt.close()


def plot_modular_return(gru_curve, ps_curve, period, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    for ax, normalized, ylabel in [
        (axes[0], False, f"Mean ||h(k+{period}) − h(k)||"),
        (axes[1], True,  f"Relative ||Δ|| / mean ||h||"),
    ]:
        for (ks, abs_, ss, rel_), color, label in [
            (gru_curve, "C3", "GRU"),
            (ps_curve,  "C0", "PhaseSumNet"),
        ]:
            y = rel_ if normalized else abs_
            ax.plot(ks, y, marker="o", color=color, lw=2, label=label)
            if not normalized:
                ax.fill_between(ks, abs_ - ss, abs_ + ss,
                                color=color, alpha=0.20)
        ax.axhline(0, color="k", lw=0.5, ls=":", label="Perfect closure")
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"Modular return — does adding {period} TWIRLs leave the state where it was?")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_pca(states, k_values, kind, title, path):
    proj, explained = project_pca(states, k_values)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    # Colour by k (k=0 red, k=max purple, etc.)
    cmap = plt.get_cmap("viridis")
    for k in k_values:
        c = cmap(k / max(k_values))
        pts = proj[k]
        ax.scatter(pts[:, 0], pts[:, 1], s=22, color=c, alpha=0.55,
                   label=f"k={k}" if k in (0, 5, 10, 15, 20, 25, 30) else None)
        # Cluster centroid
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ax.scatter([cx], [cy], s=140, color=c, marker="X",
                   edgecolors="black", linewidths=0.8)
    # Connect centroids in k-order to show trajectory of the mean.
    centroids = np.array([[proj[k][:, 0].mean(), proj[k][:, 1].mean()]
                           for k in k_values])
    ax.plot(centroids[:, 0], centroids[:, 1], color="black", lw=0.8,
            alpha=0.5, zorder=0)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    spec = CyclicTaskSpec(5)

    # ---- 1. Train fresh models, same setup as Exp 5 / Exp 4 ----
    print("\n>>> Training GRU (d=64, L=1, Z/5)")
    gru = GRUMultilayer(spec, d_model=64, n_layers=1)
    gru, _ = train_model(
        gru, spec,
        train_depth_range=(0, 5),
        eval_depths=tuple(range(0, 21)),
        n_train=60_000, n_eval_per_depth=2_000,
        batch_size=256, lr=3e-3, n_epochs=15,
        eval_every_steps=500, device=device, seed=0,
        tag="GRU-trajectory")

    print("\n>>> Training PhaseSumNet (d=16, Z/5)")
    ps = PhaseSumRef(spec, d_model=16)
    ps, _ = train_model(
        ps, spec,
        train_depth_range=(0, 5),
        eval_depths=tuple(range(0, 21)),
        n_train=60_000, n_eval_per_depth=2_000,
        batch_size=256, lr=3e-3, n_epochs=15,
        eval_every_steps=500, device=device, seed=0,
        tag="PhaseSum-trajectory")

    gru.eval(); ps.eval()

    # ---- 2. Probe sequences ----
    k_values = list(range(0, 31))
    tokens_by_k = make_probe_sequences(spec, k_values, n_samples=64)
    tokens_by_k = {k: t.to(device) for k, t in tokens_by_k.items()}

    gru_states = collect_trajectories(gru, tokens_by_k, "gru")
    ps_states  = collect_trajectories(ps,  tokens_by_k, "phasesum")

    # ---- 3. Three diagnostics ----
    # (a) magnitude
    plot_magnitude(magnitude_curve(gru_states),
                    magnitude_curve(ps_states),
                    os.path.join(OUTPUT_DIR, "magnitude.png"))

    # (b) modular return at period 5
    plot_modular_return(modular_return_curve(gru_states, 5),
                         modular_return_curve(ps_states,  5),
                         5,
                         os.path.join(OUTPUT_DIR, "modular_return.png"))

    # (b') logit-level closure — the diagnostic that ACTUALLY predicts accuracy
    gru_logit = logit_modular_return(gru, tokens_by_k, 5, device)
    ps_logit  = logit_modular_return(ps,  tokens_by_k, 5, device)
    plot_logit_closure(gru_logit, ps_logit, 5,
                        os.path.join(OUTPUT_DIR, "logit_closure.png"))

    # (c) PCA trajectories
    plot_pca(gru_states, k_values, "gru",
              "GRU hidden state trajectory (Z/5, k=0..30)",
              os.path.join(OUTPUT_DIR, "gru_pca.png"))
    plot_pca(ps_states, k_values, "phasesum",
              "PhaseSumNet [cos,sin] trajectory (Z/5, k=0..30)",
              os.path.join(OUTPUT_DIR, "phasesum_pca.png"))

    # ---- 4. Summary text ----
    summary = {}
    for name, states, model in [("GRU", gru_states, gru),
                                  ("PhaseSumNet", ps_states, ps)]:
        ks, mags, _ = magnitude_curve(states)
        rks, rdists, _, rnorm = modular_return_curve(states, 5)
        lks, ldists, linv = logit_modular_return(model, tokens_by_k, 5, device)
        summary[name] = {
            "magnitude_at_k": {int(k): float(m) for k, m in zip(ks, mags)},
            "modular_return_5_abs":      {int(k): float(d) for k, d in zip(rks, rdists)},
            "modular_return_5_relative": {int(k): float(r) for k, r in zip(rks, rnorm)},
            "logit_dist_5":              {int(k): float(d) for k, d in zip(lks, ldists)},
            "argmax_invariance_5":       {int(k): float(i) for k, i in zip(lks, linv)},
            "magnitude_growth_ratio_30_over_5": float(mags[ks.tolist().index(30)] /
                                                       mags[ks.tolist().index(5)]),
            "mean_modular_return_5_abs":       float(rdists.mean()),
            "mean_modular_return_5_relative":  float(rnorm.mean()),
            "mean_logit_dist_5":               float(ldists.mean()),
            "mean_argmax_invariance_5":        float(linv.mean()),
            "mean_magnitude_OOD":              float(mags[6:].mean()),
        }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Trajectory diagnostics (Z/5) ===")
    print("Note: state-level closure metrics can mislead. The metric that")
    print("actually predicts accuracy is ARGMAX INVARIANCE: does the model's")
    print("prediction stay the same after adding 5 TWIRLs?")
    for name, s in summary.items():
        print(f"\n{name}:")
        print(f"  state magnitude at k>5:                "
              f"{s['mean_magnitude_OOD']:.4f}")
        print(f"  state magnitude ratio  30 / 5:         "
              f"{s['magnitude_growth_ratio_30_over_5']:.4f}")
        print(f"  state ||Δ|| absolute:                  "
              f"{s['mean_modular_return_5_abs']:.4f}")
        print(f"  state ||Δ|| / ||h|| (relative):        "
              f"{s['mean_modular_return_5_relative']:.4f}")
        print(f"  logit ||Δ||:                           "
              f"{s['mean_logit_dist_5']:.4f}")
        print(f"  argmax invariance after 5·TWIRL:       "
              f"{s['mean_argmax_invariance_5']:.4f}  (target 1.0)")


if __name__ == "__main__":
    main()
