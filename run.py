"""
Run the full experiment.

What this script does, end to end:
  1. Train RealTransformer (param-matched) on depths {0..MAX_TRAIN_DEPTH}.
  2. Train ComplexTransformer on the same data.
  3. Evaluate both on depths {0..MAX_EVAL_DEPTH} (out-of-distribution is
     anything beyond MAX_TRAIN_DEPTH).
  4. Repeat for N_SEEDS independent seeds.
  5. Run the phase probe on the complex model.
  6. Save:
       results.json       all metrics
       depth_accuracy.png mean ± stderr per depth, both models
       training_curves.png training loss + eval accuracy over steps
       phase_probe.png    histogram of Δangle from NOT
       summary.txt        text summary of key statistics

Run with:
    python3 run.py --config full      # H100 run (a few hours)
    python3 run.py --config smoke     # ~1 min, CPU-OK, sanity check
"""

import argparse
import json
import os
import math
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import (RealTransformer, ComplexTransformer,
                    count_params, matched_configs)
from train import train_model
from analyze import probe_not_phase


# --------------- CONFIG ---------------

CONFIGS = {
    # Fast smoke test: verifies the whole pipeline works end-to-end.
    "smoke": dict(
        n_seeds            = 1,
        d_complex          = 32,
        n_heads            = 4,
        n_layers           = 2,
        max_train_depth    = 2,
        max_eval_depth     = 5,
        n_train            = 4_000,
        n_eval_per_depth   = 200,
        n_epochs           = 3,
        batch_size         = 128,
        lr                 = 3e-4,
        eval_every_steps   = 30,
    ),

    # Full experiment for H100. Tuned to give clean signal in <1 hour.
    "full": dict(
        n_seeds            = 5,
        d_complex          = 64,
        n_heads            = 4,
        n_layers           = 3,
        max_train_depth    = 3,
        max_eval_depth     = 10,
        n_train            = 100_000,
        n_eval_per_depth   = 2_000,
        n_epochs           = 30,
        batch_size         = 256,
        lr                 = 3e-4,
        eval_every_steps   = 200,
    ),
}

OUTPUT_DIR = "results"


# --------------- RUN ONE SEED ---------------

def run_one_seed(cfg, seed, device):
    d_complex = cfg["d_complex"]
    n_heads   = cfg["n_heads"]
    n_layers  = cfg["n_layers"]
    d_real    = matched_configs(d_complex, n_heads, n_layers)
    train_dr  = (0, cfg["max_train_depth"])
    eval_d    = tuple(range(0, cfg["max_eval_depth"] + 1))

    real_model = RealTransformer(d_model=d_real,    n_heads=n_heads, n_layers=n_layers)
    cplx_model = ComplexTransformer(d_model=d_complex, n_heads=n_heads, n_layers=n_layers)

    real_params = count_params(real_model)
    cplx_params = count_params(cplx_model)
    print(f"\n=== Seed {seed}  (real d={d_real}: {real_params:,} params  |  "
          f"complex d={d_complex}: {cplx_params:,} params) ===")

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

    real_model, real_hist = train_model(real_model, tag=f"real-s{seed}",  **common)
    cplx_model, cplx_hist = train_model(cplx_model, tag=f"cplx-s{seed}",  **common)

    # Phase probe on the complex model.
    delta, summary = probe_not_phase(cplx_model, device=device, seed=seed)
    print(f"  Phase probe: mean|Δ|={summary['mean_abs_rad']:.3f} rad  "
          f"(π={math.pi:.3f})  frac near π = {summary['frac_near_pi']:.2f}")

    return {
        "seed": seed,
        "real_params": real_params,
        "cplx_params": cplx_params,
        "real_history": real_hist,
        "cplx_history": cplx_hist,
        "phase_probe":  summary,
        "phase_deltas": delta.tolist(),
    }


# --------------- PLOTTING ---------------

def aggregate(seed_results, key):
    """Stack metrics across seeds for averaging. Each seed contributes one
       dict {depth: acc}; we produce mean/stderr per depth."""
    all_depths = sorted({
        d for r in seed_results for d in r[key]["final_by_depth"]
    })
    M = np.array([
        [r[key]["final_by_depth"][d] for d in all_depths]
        for r in seed_results
    ])
    return np.array(all_depths), M.mean(axis=0), M.std(axis=0) / math.sqrt(len(seed_results))


def plot_depth_accuracy(seed_results, cfg, path):
    real_results = [{"final_by_depth": r["real_history"]["final_by_depth"]} for r in seed_results]
    cplx_results = [{"final_by_depth": r["cplx_history"]["final_by_depth"]} for r in seed_results]

    # Pull out arrays.
    depths = sorted({d for r in real_results for d in r["final_by_depth"]})
    R = np.array([[r["final_by_depth"][d] for d in depths] for r in real_results])
    C = np.array([[r["final_by_depth"][d] for d in depths] for r in cplx_results])
    n = R.shape[0]
    rm, rs = R.mean(0), R.std(0) / math.sqrt(n)
    cm, cs = C.mean(0), C.std(0) / math.sqrt(n)

    plt.figure(figsize=(7, 4.5))
    plt.fill_between(depths, rm - rs, rm + rs, alpha=0.20, color="C3")
    plt.fill_between(depths, cm - cs, cm + cs, alpha=0.20, color="C0")
    plt.plot(depths, rm, "o-", color="C3", label="Real transformer")
    plt.plot(depths, cm, "s-", color="C0", label="Complex transformer")
    plt.axvspan(-0.5, cfg["max_train_depth"] + 0.5, color="gray", alpha=0.10,
                label="Train distribution")
    plt.axhline(0.5, color="k", lw=0.5, ls=":", label="Chance")
    plt.xlabel("Negation depth (# of NOT tokens)")
    plt.ylabel("Accuracy")
    plt.title("Depth generalization: in-distribution vs out-of-distribution")
    plt.ylim(-0.02, 1.02)
    plt.xticks(depths)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_training_curves(seed_results, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in seed_results:
        steps = r["real_history"]["steps"]
        axes[0].plot(steps, r["real_history"]["train_loss"], color="C3", alpha=0.5)
        axes[0].plot(steps, r["cplx_history"]["train_loss"], color="C0", alpha=0.5)
        axes[1].plot(steps, r["real_history"]["eval_acc"],   color="C3", alpha=0.5)
        axes[1].plot(steps, r["cplx_history"]["eval_acc"],   color="C0", alpha=0.5)
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Train loss"); axes[0].set_title("Training loss")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Eval accuracy"); axes[1].set_title("Eval accuracy (all depths)")
    axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
    axes[0].plot([], [], color="C3", label="Real")
    axes[0].plot([], [], color="C0", label="Complex")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_phase_probe(seed_results, path):
    plt.figure(figsize=(7, 4))
    all_deltas = []
    for r in seed_results:
        all_deltas.extend(r["phase_deltas"])
    plt.hist([abs(x) for x in all_deltas], bins=40, color="C0", edgecolor="white")
    plt.axvline(math.pi, color="red", ls="--", label="π (perfect phase flip)")
    plt.xlabel("|Δ angle of CLS readout| after one extra NOT  (radians)")
    plt.ylabel("Count")
    plt.title("Mechanistic probe: does NOT act as a π rotation?")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# --------------- ENTRY POINT ---------------

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

    # -------- aggregate & save --------
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        slim = []
        for r in seed_results:
            slim.append({
                "seed": r["seed"],
                "real_params": r["real_params"],
                "cplx_params": r["cplx_params"],
                "real_final_acc": r["real_history"]["final_acc"],
                "cplx_final_acc": r["cplx_history"]["final_acc"],
                "real_by_depth": r["real_history"]["final_by_depth"],
                "cplx_by_depth": r["cplx_history"]["final_by_depth"],
                "phase_probe":   r["phase_probe"],
            })
        json.dump({"config": cfg, "results": slim}, f, indent=2)

    plot_depth_accuracy (seed_results, cfg, os.path.join(args.outdir, "depth_accuracy.png"))
    plot_training_curves(seed_results,      os.path.join(args.outdir, "training_curves.png"))
    plot_phase_probe    (seed_results,      os.path.join(args.outdir, "phase_probe.png"))

    # -------- text summary --------
    def stat(values):
        a = np.array(values)
        return f"{a.mean():.3f} ± {a.std()/math.sqrt(len(a)):.3f}"

    real_in, cplx_in   = [], []
    real_out, cplx_out = [], []
    max_train = cfg["max_train_depth"]
    for r in seed_results:
        for d, a in r["real_history"]["final_by_depth"].items():
            (real_in if d <= max_train else real_out).append(a)
        for d, a in r["cplx_history"]["final_by_depth"].items():
            (cplx_in if d <= max_train else cplx_out).append(a)

    phase_means = [r["phase_probe"]["mean_abs_rad"] for r in seed_results]
    phase_nears = [r["phase_probe"]["frac_near_pi"] for r in seed_results]

    lines = [
        f"Configuration: {args.config}",
        f"Seeds: {cfg['n_seeds']}",
        f"Train depths: 0..{cfg['max_train_depth']}    "
        f"Eval depths: 0..{cfg['max_eval_depth']}",
        "",
        f"Real model:    in-distribution acc = {stat(real_in)}    "
        f"out-of-distribution acc = {stat(real_out)}",
        f"Complex model: in-distribution acc = {stat(cplx_in)}    "
        f"out-of-distribution acc = {stat(cplx_out)}",
        "",
        f"OOD gap (Complex − Real): "
        f"{np.mean(cplx_out) - np.mean(real_out):+.3f}",
        "",
        f"Phase probe on complex model:",
        f"  mean |Δ| angle from one extra NOT = {stat(phase_means)} rad   "
        f"(π = {math.pi:.3f})",
        f"  fraction of pairs within π/8 of π = {stat(phase_nears)}",
        "",
        f"Wallclock: {dt/60:.1f} min",
    ]
    summary = "\n".join(lines)
    print("\n" + "=" * 60)
    print(summary)
    print("=" * 60)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
