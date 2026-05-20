"""
Experiment 3: rotation mod 3.

Tests the four architectures defined in models_triple.py on the mod-3
parity task. The central prediction:

    RealAddNet (additive + linear readout) cannot solve mod-3 at OOD
    depths, by theorem. The other three architectures should.

If observed, this is the cleanest possible separation: it's not 'complex
vs real' (both real and complex architectures with multiplicative
composition succeed), it's 'multiplicative or periodic vs purely
additive-linear'.

Run on H100:
    python3 run_triple.py --config full
Smoke test:
    python3 run_triple.py --config smoke
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

from models_triple import (PhaseSumNet3, RealAddNet, GatedComplexRNN3,
                            GRUBaseline3, count_params, matched_d_gru)
from train_triple import train_model
from analyze_triple import (probe_phase_sum_net3, probe_real_add_net,
                             probe_complex_rnn3_behavioral)

CONFIGS = {
    "smoke": dict(
        n_seeds=1, d_phase_sum=8, d_complex_rnn=16,
        max_train_depth=5, max_eval_depth=10,
        n_train=6_000, n_eval_per_depth=300, n_epochs=3,
        batch_size=128, lr=1e-2, eval_every_steps=30,
    ),
    "full": dict(
        n_seeds=5, d_phase_sum=16, d_complex_rnn=32,
        max_train_depth=5, max_eval_depth=20,
        n_train=60_000, n_eval_per_depth=2_000, n_epochs=15,
        batch_size=256, lr=5e-3, eval_every_steps=200,
    ),
}

OUTPUT_DIR = "results_exp3"


def run_one_seed(cfg, seed, device):
    d_ps   = cfg["d_phase_sum"]
    d_cplx = cfg["d_complex_rnn"]
    d_gru  = matched_d_gru(d_cplx)
    train_dr = (0, cfg["max_train_depth"])
    eval_d   = tuple(range(0, cfg["max_eval_depth"] + 1))

    models = {
        "PhaseSumNet3":     PhaseSumNet3(d_model=d_ps),
        "RealAddNet":       RealAddNet(d_model=d_ps),
        "GatedComplexRNN3": GatedComplexRNN3(d_model=d_cplx),
        "GRUBaseline3":     GRUBaseline3(d_model=d_gru),
    }
    params = {k: count_params(v) for k, v in models.items()}

    print(f"\n=== Seed {seed} ===")
    for k, n in params.items():
        print(f"  {k:20s}  params={n:>6,}")

    common = dict(
        train_depth_range=train_dr, eval_depths=eval_d,
        n_train=cfg["n_train"], n_eval_per_depth=cfg["n_eval_per_depth"],
        batch_size=cfg["batch_size"], lr=cfg["lr"],
        n_epochs=cfg["n_epochs"], eval_every_steps=cfg["eval_every_steps"],
        device=device, seed=seed,
    )

    histories = {}
    for name, m in models.items():
        trained, hist = train_model(m, tag=f"{name}-s{seed}", **common)
        models[name] = trained
        histories[name] = hist

    probes = {
        "PhaseSumNet3":     probe_phase_sum_net3(models["PhaseSumNet3"]),
        "RealAddNet":       probe_real_add_net (models["RealAddNet"]),
        "GatedComplexRNN3": probe_complex_rnn3_behavioral(
                                models["GatedComplexRNN3"], device),
        "GRUBaseline3":     probe_complex_rnn3_behavioral(
                                models["GRUBaseline3"], device),
    }

    print(f"\n  [Probe] PhaseSumNet3 cos(3·θ(TWIRL))="
          f"{probes['PhaseSumNet3']['cycle3_cos_mean']:+.3f}  "
          f"frac near 1 = {probes['PhaseSumNet3']['frac_cycle3_near_one']:.2f}")
    print(f"  [Probe] RealAddNet slope-spread = "
          f"{probes['RealAddNet']['slope_spread']:.3f}  "
          f"dominant class at large k = "
          f"{probes['RealAddNet']['dominant_class_at_large_k']}")
    print(f"  [Probe] GatedComplexRNN3 closure under 3·TWIRL = "
          f"{probes['GatedComplexRNN3']['frac_invariant_under_3_twirls']:.3f}")
    print(f"  [Probe] GRUBaseline3      closure under 3·TWIRL = "
          f"{probes['GRUBaseline3']['frac_invariant_under_3_twirls']:.3f}")

    return {"seed": seed, "params": params,
            "histories": histories, "probes": probes}


COLORS  = {"PhaseSumNet3":"C0", "RealAddNet":"C1",
            "GatedComplexRNN3":"C2", "GRUBaseline3":"C3"}
MARKERS = {"PhaseSumNet3":"s",  "RealAddNet":"D",
            "GatedComplexRNN3":"^", "GRUBaseline3":"o"}


def plot_depth_accuracy(seed_results, cfg, path):
    names = list(seed_results[0]["histories"])
    depths = sorted({d for r in seed_results
                     for n in names
                     for d in r["histories"][n]["final_by_depth"]})

    plt.figure(figsize=(9, 5.2))
    for name in names:
        arr = np.array([[r["histories"][name]["final_by_depth"][d]
                         for d in depths] for r in seed_results])
        mean = arr.mean(0)
        se = arr.std(0) / math.sqrt(arr.shape[0])
        c, m = COLORS[name], MARKERS[name]
        plt.fill_between(depths, mean - se, mean + se, alpha=0.18, color=c)
        plt.plot(depths, mean, marker=m, color=c, lw=2,
                 label=f"{name} ({seed_results[0]['params'][name]:,} params)")
    plt.axvspan(-0.5, cfg["max_train_depth"] + 0.5, color="gray", alpha=0.10,
                label="Train distribution")
    plt.axhline(1.0 / 3, color="k", lw=0.5, ls=":", label="Chance (1/3)")
    plt.xlabel("Twirl depth k  (# of TWIRL tokens)")
    plt.ylabel("Accuracy (3-class)")
    plt.title("Experiment 3 — mod-3 rotation:\nadditive-linear cannot generalize OOD")
    plt.ylim(-0.02, 1.05)
    plt.xticks(depths)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=140); plt.close()


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
    axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3); axes[0].legend()
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_phase_per_dim(seed_results, path):
    """For PhaseSumNet3: histogram of cos(3·θ(TWIRL)) per dim.

    A successful model has values near +1. A failed model is spread."""
    vals = []
    for r in seed_results:
        for theta in r["probes"]["PhaseSumNet3"]["phase_twirl_per_dim"]:
            vals.append(math.cos(3 * theta))
    plt.figure(figsize=(7.5, 4))
    plt.hist(vals, bins=30, color="C0", edgecolor="white")
    plt.axvline(1.0, color="red", ls="--", lw=2, label="+1 (3·θ ≡ 0 mod 2π)")
    plt.xlabel("cos(3·θ(TWIRL)) per dimension")
    plt.ylabel("count")
    plt.title("PhaseSumNet3: does 3 TWIRLs return phase to identity?")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=140); plt.close()


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

    slim = []
    for r in seed_results:
        slim.append({
            "seed": r["seed"], "params": r["params"],
            "final_acc":      {n: r["histories"][n]["final_acc"]
                                for n in r["histories"]},
            "final_by_depth": {n: r["histories"][n]["final_by_depth"]
                                for n in r["histories"]},
            "probes": r["probes"],
        })
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump({"config": cfg, "results": slim}, f, indent=2)

    plot_depth_accuracy (seed_results, cfg, os.path.join(args.outdir, "depth_accuracy.png"))
    plot_training_curves(seed_results,      os.path.join(args.outdir, "training_curves.png"))
    plot_phase_per_dim  (seed_results,      os.path.join(args.outdir, "phase_per_dim.png"))

    def stat(values):
        a = np.array(values)
        if len(a) == 0: return "n/a"
        return f"{a.mean():.3f} ± {a.std() / math.sqrt(len(a)):.3f}"

    names = list(seed_results[0]["histories"])
    max_train = cfg["max_train_depth"]
    lines = [
        f"Configuration: {args.config}",
        f"Seeds: {cfg['n_seeds']}   Train depths: 0..{max_train}   "
        f"Eval depths: 0..{cfg['max_eval_depth']}",
        "",
        f"{'model':20s}  {'params':>8s}  {'ID acc':>14s}  {'OOD acc':>14s}",
        "-" * 62,
    ]
    for name in names:
        in_acc, out_acc = [], []
        for r in seed_results:
            for d, a in r["histories"][name]["final_by_depth"].items():
                (in_acc if d <= max_train else out_acc).append(a)
        params = seed_results[0]["params"][name]
        lines.append(f"{name:20s}  {params:>8,}  {stat(in_acc):>14s}  {stat(out_acc):>14s}")

    # Per-depth OOD breakdown — most informative for the negative control.
    lines += ["", "Per-depth OOD accuracy (mean across seeds):"]
    depths_sorted = sorted(seed_results[0]["histories"][names[0]]["final_by_depth"])
    ood_depths = [d for d in depths_sorted if d > max_train]
    header = "  depth  " + "  ".join(f"{n[:14]:>14s}" for n in names)
    lines.append(header)
    for d in ood_depths:
        row = f"  {d:>5d}  "
        for n in names:
            accs = [r["histories"][n]["final_by_depth"][d] for r in seed_results]
            row += f"{np.mean(accs):>14.3f}  "
        lines.append(row)

    lines += ["", "Probes (means across seeds):"]
    ps_cycle = [r["probes"]["PhaseSumNet3"]["cycle3_cos_mean"]
                for r in seed_results]
    ps_frac  = [r["probes"]["PhaseSumNet3"]["frac_cycle3_near_one"]
                for r in seed_results]
    lines += [
        f"  PhaseSumNet3 — mod-3 closure check:",
        f"    cos(3·θ(TWIRL)) = {stat(ps_cycle)}     (target: +1)",
        f"    frac of dims with cos(3·θ)>0.9 = {stat(ps_frac)}",
    ]
    ra_spread = [r["probes"]["RealAddNet"]["slope_spread"] for r in seed_results]
    lines += [
        f"  RealAddNet — readout-slope analysis (the failure signature):",
        f"    slope spread max-min = {stat(ra_spread)}",
        f"    (a non-zero spread predicts which class dominates at large k)",
    ]
    cr_inv = [r["probes"]["GatedComplexRNN3"]["frac_invariant_under_3_twirls"]
              for r in seed_results]
    gr_inv = [r["probes"]["GRUBaseline3"]["frac_invariant_under_3_twirls"]
              for r in seed_results]
    lines += [
        f"  GatedComplexRNN3 — closure under 3·TWIRL: {stat(cr_inv)}  (target: 1.0)",
        f"  GRUBaseline3      — closure under 3·TWIRL: {stat(gr_inv)}  (target: 1.0)",
        "", f"Wallclock: {dt/60:.1f} min",
    ]
    summary = "\n".join(lines)
    print("\n" + "=" * 64); print(summary); print("=" * 64)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
