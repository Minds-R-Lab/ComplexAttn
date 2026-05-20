"""
Experiment 2: multiplicative composition for the parity-of-negation task.

Trains three architectures on the same task as Experiment 1:
  - PhaseSumNet      (minimal, set-equivariant, complex multiplicative)
  - GatedComplexRNN  (bidirectional gated complex RNN)
  - GRUBaseline      (standard bidirectional GRU, parameter-matched)

Evaluates depth generalization (train on depths 0..D_train, test on
depths 0..D_eval where D_eval ≫ D_train), runs a direct mechanistic
probe (we can read the learned phase for NOT directly), and produces
plots + a summary.

Run on H100:
    python3 run_rnn.py --config full
Quick smoke test:
    python3 run_rnn.py --config smoke
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rnn_models import (PhaseSumNet, GatedComplexRNN, GRUBaseline,
                         count_params, matched_d_gru)
from train import train_model
from analyze_rnn import probe_phase_sum_net, probe_gated_complex_rnn


# ---------------- CONFIGS ----------------

CONFIGS = {
    "smoke": dict(
        n_seeds          = 1,
        d_phase_sum      = 8,
        d_complex_rnn    = 16,
        max_train_depth  = 2,
        max_eval_depth   = 6,
        n_train          = 4_000,
        n_eval_per_depth = 200,
        n_epochs         = 3,
        batch_size       = 128,
        lr               = 1e-2,
        eval_every_steps = 30,
    ),
    "full": dict(
        n_seeds          = 5,
        d_phase_sum      = 16,
        d_complex_rnn    = 32,
        max_train_depth  = 3,
        max_eval_depth   = 15,        # push OOD generalization HARD
        n_train          = 60_000,
        n_eval_per_depth = 2_000,
        n_epochs         = 20,
        batch_size       = 256,
        lr               = 5e-3,
        eval_every_steps = 200,
    ),
}

OUTPUT_DIR = "results_exp2"


# ---------------- RUN ONE SEED ----------------

def run_one_seed(cfg, seed, device):
    d_ps   = cfg["d_phase_sum"]
    d_cplx = cfg["d_complex_rnn"]
    d_gru  = matched_d_gru(d_cplx)
    train_dr = (0, cfg["max_train_depth"])
    eval_d   = tuple(range(0, cfg["max_eval_depth"] + 1))

    models = {
        "PhaseSumNet":     PhaseSumNet(d_model=d_ps),
        "GatedComplexRNN": GatedComplexRNN(d_model=d_cplx),
        "GRUBaseline":     GRUBaseline(d_model=d_gru),
    }
    params = {k: count_params(v) for k, v in models.items()}

    print(f"\n=== Seed {seed} ===")
    for k, n in params.items():
        print(f"  {k:18s}  params={n:>6,}")

    common = dict(
        train_depth_range = train_dr,
        eval_depths       = eval_d,
        n_train           = cfg["n_train"],
        n_eval_per_depth  = cfg["n_eval_per_depth"],
        batch_size        = cfg["batch_size"],
        lr                = cfg["lr"],
        n_epochs          = cfg["n_epochs"],
        eval_every_steps  = cfg["eval_every_steps"],
        device            = device,
        seed              = seed,
    )

    histories = {}
    for name, m in models.items():
        trained, hist = train_model(m, tag=f"{name}-s{seed}", **common)
        models[name] = trained
        histories[name] = hist

    probes = {
        "PhaseSumNet":     probe_phase_sum_net(models["PhaseSumNet"]),
        "GatedComplexRNN": probe_gated_complex_rnn(models["GatedComplexRNN"]),
    }
    print(f"\n  [Probe] PhaseSumNet  cos(NOT)={probes['PhaseSumNet']['cos_not_mean']:+.3f}  "
          f"frac NOT near π = {probes['PhaseSumNet']['frac_not_near_pi']:.2f}")
    print(f"  [Probe] GatedComplexRNN fwd cos(NOT)={probes['GatedComplexRNN']['fwd_cos_not_mean']:+.3f}  "
          f"bwd cos(NOT)={probes['GatedComplexRNN']['bwd_cos_not_mean']:+.3f}")

    return {
        "seed":      seed,
        "params":    params,
        "histories": histories,
        "probes":    probes,
    }


# ---------------- PLOTTING ----------------

COLORS = {
    "PhaseSumNet":     "C0",
    "GatedComplexRNN": "C2",
    "GRUBaseline":     "C3",
}
MARKERS = {
    "PhaseSumNet":     "s",
    "GatedComplexRNN": "^",
    "GRUBaseline":     "o",
}


def plot_depth_accuracy(seed_results, cfg, path):
    names = list(seed_results[0]["histories"])
    depths = sorted({d for r in seed_results
                     for n in names
                     for d in r["histories"][n]["final_by_depth"]})

    plt.figure(figsize=(8.5, 5))
    for name in names:
        arr = np.array([[r["histories"][name]["final_by_depth"][d]
                         for d in depths] for r in seed_results])
        mean = arr.mean(0)
        se   = arr.std(0) / math.sqrt(arr.shape[0])
        c, m = COLORS[name], MARKERS[name]
        plt.fill_between(depths, mean - se, mean + se, alpha=0.18, color=c)
        plt.plot(depths, mean, marker=m, color=c, lw=2,
                 label=f"{name} ({seed_results[0]['params'][name]:,} params)")
    plt.axvspan(-0.5, cfg["max_train_depth"] + 0.5, color="gray", alpha=0.10,
                label="Train distribution")
    plt.axhline(0.5, color="k", lw=0.5, ls=":", label="Chance")
    plt.xlabel("Negation depth (# of NOT tokens)")
    plt.ylabel("Accuracy")
    plt.title("Experiment 2: depth generalization with multiplicative composition")
    plt.ylim(-0.02, 1.05)
    plt.xticks(depths)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_training_curves(seed_results, path):
    names = list(seed_results[0]["histories"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in seed_results:
        for name in names:
            steps = r["histories"][name]["steps"]
            axes[0].plot(steps, r["histories"][name]["train_loss"],
                         color=COLORS[name], alpha=0.5)
            axes[1].plot(steps, r["histories"][name]["eval_acc"],
                         color=COLORS[name], alpha=0.5)
    for name in names:
        axes[0].plot([], [], color=COLORS[name], label=name)
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Train loss")
    axes[0].set_title("Training loss")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Eval accuracy (all depths)")
    axes[1].set_title("Eval accuracy over training")
    axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_phase_per_dim(seed_results, path):
    """Histogram of learned NOT-phase per dimension across all seeds."""
    all_phases = []
    for r in seed_results:
        all_phases.extend(r["probes"]["PhaseSumNet"]["phase_not_per_dim"])
    plt.figure(figsize=(7.5, 4))
    plt.hist([abs(p) for p in all_phases], bins=30, color="C0", edgecolor="white")
    plt.axvline(math.pi, color="red", ls="--", lw=2, label="π (predicted)")
    plt.xlabel("|learned phase for NOT| per dimension  (radians)")
    plt.ylabel("count")
    plt.title("Did PhaseSumNet learn |θ(NOT)| ≈ π per dimension?")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# ---------------- ENTRY ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full", choices=list(CONFIGS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=OUTPUT_DIR)
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Configuration: {args.config}")
    print(json.dumps(cfg, indent=2))
    print(f"Device: {device}")

    t0 = time.time()
    seed_results = []
    for seed in range(cfg["n_seeds"]):
        seed_results.append(run_one_seed(cfg, seed=seed, device=device))
    dt = time.time() - t0
    print(f"\nTotal wallclock: {dt/60:.1f} min")

    # --- Save raw metrics ---
    slim = []
    for r in seed_results:
        slim.append({
            "seed":   r["seed"],
            "params": r["params"],
            "final_acc":      {n: r["histories"][n]["final_acc"]
                                for n in r["histories"]},
            "final_by_depth": {n: r["histories"][n]["final_by_depth"]
                                for n in r["histories"]},
            "probes":         r["probes"],
        })
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump({"config": cfg, "results": slim}, f, indent=2)

    plot_depth_accuracy (seed_results, cfg, os.path.join(args.outdir, "depth_accuracy.png"))
    plot_training_curves(seed_results,      os.path.join(args.outdir, "training_curves.png"))
    plot_phase_per_dim  (seed_results,      os.path.join(args.outdir, "phase_per_dim.png"))

    # --- Text summary ---
    def stat(values):
        a = np.array(values)
        if len(a) == 0:
            return "n/a"
        return f"{a.mean():.3f} ± {a.std() / math.sqrt(len(a)):.3f}"

    names = list(seed_results[0]["histories"])
    max_train = cfg["max_train_depth"]

    lines = [
        f"Configuration: {args.config}",
        f"Seeds: {cfg['n_seeds']}    "
        f"Train depths: 0..{cfg['max_train_depth']}    "
        f"Eval depths: 0..{cfg['max_eval_depth']}",
        "",
        f"{'model':18s}  {'params':>8s}  {'ID acc':>14s}  {'OOD acc':>14s}",
        "-" * 60,
    ]
    for name in names:
        in_acc, out_acc = [], []
        for r in seed_results:
            for d, a in r["histories"][name]["final_by_depth"].items():
                (in_acc if d <= max_train else out_acc).append(a)
        params = seed_results[0]["params"][name]
        lines.append(f"{name:18s}  {params:>8,}  {stat(in_acc):>14s}  {stat(out_acc):>14s}")

    # Probe summary
    lines += ["", "Mechanistic probes (means across seeds):"]
    ps_cos_not = [r["probes"]["PhaseSumNet"]["cos_not_mean"]   for r in seed_results]
    ps_cos_t   = [r["probes"]["PhaseSumNet"]["cos_t_mean"]     for r in seed_results]
    ps_cos_f   = [r["probes"]["PhaseSumNet"]["cos_f_mean"]     for r in seed_results]
    ps_cos_fl  = [r["probes"]["PhaseSumNet"]["cos_filler_mean"] for r in seed_results]
    ps_near_pi = [r["probes"]["PhaseSumNet"]["frac_not_near_pi"] for r in seed_results]
    lines += [
        f"  PhaseSumNet — cos(phase) per token class (target sign):",
        f"    cos(NOT)    = {stat(ps_cos_not)}     (target: −1)",
        f"    cos(T)      = {stat(ps_cos_t)}     (target: +1)",
        f"    cos(F)      = {stat(ps_cos_f)}     (target: −1)",
        f"    cos(filler) = {stat(ps_cos_fl)}     (target: +1)",
        f"    frac of NOT dims within π/8 of ±π = {stat(ps_near_pi)}",
    ]
    rnn_fwd = [r["probes"]["GatedComplexRNN"]["fwd_cos_not_mean"] for r in seed_results]
    rnn_bwd = [r["probes"]["GatedComplexRNN"]["bwd_cos_not_mean"] for r in seed_results]
    rnn_fwd_near = [r["probes"]["GatedComplexRNN"]["fwd_frac_not_near_pi"] for r in seed_results]
    rnn_bwd_near = [r["probes"]["GatedComplexRNN"]["bwd_frac_not_near_pi"] for r in seed_results]
    lines += [
        f"  GatedComplexRNN — cos(phase(NOT)) per direction:",
        f"    fwd cos(NOT) = {stat(rnn_fwd)}  near π: {stat(rnn_fwd_near)}",
        f"    bwd cos(NOT) = {stat(rnn_bwd)}  near π: {stat(rnn_bwd_near)}",
        "",
        f"Wallclock: {dt/60:.1f} min",
    ]
    summary = "\n".join(lines)
    print("\n" + "=" * 64)
    print(summary)
    print("=" * 64)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
