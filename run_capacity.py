"""
Experiment 5: capacity sweep for the GRU collapse at n >= 5.

Goal: distinguish "structural barrier" from "sample-inefficient".
Exp 4 found that the GRU at d=17 hits OOD 0.24 on Z/5 and Z/7,
while PhaseSumNet at d=16 hits 1.000. Two competing explanations:

  (A) Structural: real-valued gated dynamics cannot natively produce
      limit cycles of period >= 5 in tanh state space, regardless of
      capacity. Scaling won't help.

  (B) Sample-inefficient: the GRU CAN represent these groups but
      needs more capacity / data / training to find the right
      configuration. Scaling will help.

We sweep:
  - d_model in {16, 32, 64, 128, 256}     (16x range)
  - n_layers in {1, 2}                     (stacked GRU/LSTM)
  - train budget: n_train and n_epochs scaled with capacity to keep
    the chance of finding a good local minimum reasonable
  - architectures: GRU (multilayer), LSTM (multilayer), PhaseSumRef
    at matched scale
  - n in {5, 7}                            (the regime where Exp 4 found
                                             the collapse; replicates
                                             the result across more
                                             than one n)

LSTM is included as a diagnostic. If LSTM solves what GRU can't, the
mechanism is GRU-specific. If LSTM also collapses, the mechanism is
"real-valued gating can't produce arbitrary tanh limit cycles" -- a
stronger and more interesting claim.

Run on H100:
    python3 run_capacity.py --config full
Smoke:
    python3 run_capacity.py --config smoke
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

from data_cyclic import CyclicTaskSpec
from models_capacity import (GRUMultilayer, LSTMMultilayer, PhaseSumRef,
                              count_params)
from train_cyclic import train_model
from analyze_cyclic import probe_closure_under_n_twirls


# (d_model, n_layers) rungs. PhaseSumRef ignores n_layers.
CONFIGS = {
    "smoke": dict(
        n_values=(5,),
        rungs=((16, 1), (32, 1)),
        n_seeds=1,
        max_train_depth=5, max_eval_depth=10,
        n_train=6_000, n_eval_per_depth=200,
        n_epochs=3, batch_size=128, lr=5e-3, eval_every_steps=30,
    ),
    "full": dict(
        n_values=(5, 7),
        rungs=((16, 1), (32, 1), (32, 2), (64, 1), (64, 2),
                (128, 1), (128, 2), (256, 1)),
        n_seeds=3,
        max_train_depth=5, max_eval_depth=20,
        # Larger models get more samples and epochs. We don't want a
        # negative result attributable to undertraining.
        n_train=120_000,
        n_eval_per_depth=2_000,
        n_epochs=25,
        batch_size=256, lr=3e-3, eval_every_steps=300,
    ),
}

OUTPUT_DIR = "results_exp5"

ARCHITECTURES = ("GRU", "LSTM", "PhaseSum")
COLORS  = {"GRU":"C3", "LSTM":"C4", "PhaseSum":"C0"}
MARKERS = {"GRU":"o",  "LSTM":"v",  "PhaseSum":"s"}


def build(arch_name, spec, d_model, n_layers):
    if arch_name == "GRU":
        return GRUMultilayer (spec, d_model=d_model, n_layers=n_layers)
    if arch_name == "LSTM":
        return LSTMMultilayer(spec, d_model=d_model, n_layers=n_layers)
    if arch_name == "PhaseSum":
        # PhaseSum doesn't have a depth dimension; one rung per d.
        return PhaseSumRef(spec, d_model=d_model)
    raise ValueError(arch_name)


def run_one(arch_name, spec, d_model, n_layers, seed, cfg, device):
    model = build(arch_name, spec, d_model, n_layers)
    params = count_params(model)
    tag = f"{arch_name}-d{d_model}-L{n_layers}-n{spec.n}-s{seed}"

    trained, hist = train_model(
        model, spec,
        train_depth_range = (0, cfg["max_train_depth"]),
        eval_depths       = tuple(range(0, cfg["max_eval_depth"] + 1)),
        n_train           = cfg["n_train"],
        n_eval_per_depth  = cfg["n_eval_per_depth"],
        batch_size        = cfg["batch_size"],
        lr                = cfg["lr"],
        n_epochs          = cfg["n_epochs"],
        eval_every_steps  = cfg["eval_every_steps"],
        device            = device,
        seed              = seed,
        tag               = tag,
    )

    closure = probe_closure_under_n_twirls(trained, spec, device)

    fbd = hist["final_by_depth"]
    ind = [a for d, a in fbd.items() if d <= cfg["max_train_depth"]]
    ood = [a for d, a in fbd.items() if d >  cfg["max_train_depth"]]
    return {
        "arch": arch_name, "d_model": d_model, "n_layers": n_layers,
        "n": spec.n, "seed": seed, "params": params,
        "id_acc":  float(np.mean(ind)),
        "ood_acc": float(np.mean(ood)),
        "closure": closure["frac_invariant_under_n_twirls"],
        "final_by_depth": fbd,
        "wallclock_sec":  hist["wallclock_sec"],
    }


def aggregate(rows):
    """Group rows by (arch, d, L, n) -> list of per-seed metrics."""
    by_cell = {}
    for r in rows:
        # PhaseSum ignores n_layers; collapse to a single key.
        L = 1 if r["arch"] == "PhaseSum" else r["n_layers"]
        key = (r["arch"], r["d_model"], L, r["n"])
        by_cell.setdefault(key, []).append(r)
    out = {}
    for key, lst in by_cell.items():
        params = lst[0]["params"]
        ids    = np.array([x["id_acc"]  for x in lst])
        oods   = np.array([x["ood_acc"] for x in lst])
        cls    = np.array([x["closure"] for x in lst])
        out[key] = {
            "params":      params,
            "id_mean":     float(ids.mean()),
            "id_se":       float(ids.std() / math.sqrt(len(ids))),
            "ood_mean":    float(oods.mean()),
            "ood_se":      float(oods.std() / math.sqrt(len(oods))),
            "closure_mean":float(cls.mean()),
            "closure_se":  float(cls.std() / math.sqrt(len(cls))),
            "n_seeds":     len(lst),
        }
    return out


def plot_capacity_curves(agg, cfg, path):
    """For each n_value, plot OOD accuracy vs parameter count for each
       architecture. Log-x params, linear-y accuracy. PhaseSumNet
       reference line at the top should sit at 1.0 across the range."""
    ns = sorted(cfg["n_values"])
    fig, axes = plt.subplots(1, len(ns), figsize=(6 * len(ns), 5),
                              squeeze=False, sharey=True)
    for ax, n in zip(axes[0], ns):
        for arch in ARCHITECTURES:
            xs, ys, errs = [], [], []
            for (a, d, L, nn_), v in sorted(agg.items()):
                if a != arch or nn_ != n:
                    continue
                xs.append(v["params"])
                ys.append(v["ood_mean"])
                errs.append(v["ood_se"])
            if not xs:
                continue
            order = np.argsort(xs)
            xs   = np.array(xs)[order]
            ys   = np.array(ys)[order]
            errs = np.array(errs)[order]
            ax.fill_between(xs, ys - errs, ys + errs,
                            color=COLORS[arch], alpha=0.18)
            ax.plot(xs, ys, marker=MARKERS[arch], color=COLORS[arch],
                    lw=2, label=arch)
        ax.axhline(1.0 / n, color="k", lw=0.7, ls=":",
                   label=f"chance (1/{n})")
        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_title(f"Z/{n}: OOD accuracy vs capacity")
        ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="lower right", fontsize=9)
    axes[0][0].set_ylabel("Out-of-distribution accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=140); plt.close()


def plot_closure_curves(agg, cfg, path):
    """Same layout for the closure-under-n probe."""
    ns = sorted(cfg["n_values"])
    fig, axes = plt.subplots(1, len(ns), figsize=(6 * len(ns), 5),
                              squeeze=False, sharey=True)
    for ax, n in zip(axes[0], ns):
        for arch in ARCHITECTURES:
            xs, ys, errs = [], [], []
            for (a, d, L, nn_), v in sorted(agg.items()):
                if a != arch or nn_ != n:
                    continue
                xs.append(v["params"]); ys.append(v["closure_mean"])
                errs.append(v["closure_se"])
            if not xs:
                continue
            order = np.argsort(xs)
            xs, ys, errs = np.array(xs)[order], np.array(ys)[order], np.array(errs)[order]
            ax.fill_between(xs, ys - errs, ys + errs,
                            color=COLORS[arch], alpha=0.18)
            ax.plot(xs, ys, marker=MARKERS[arch], color=COLORS[arch],
                    lw=2, label=arch)
        ax.set_xscale("log"); ax.set_xlabel("Parameters")
        ax.set_title(f"Z/{n}: closure under {n}·TWIRL")
        ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="lower right", fontsize=9)
    axes[0][0].set_ylabel("Fraction invariant under n·TWIRL")
    plt.tight_layout()
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
    print(json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in cfg.items()}, indent=2, default=str))
    print(f"Device: {device}")

    # Enumerate experiments. PhaseSum ignores n_layers: dedup.
    plan = []
    for n in cfg["n_values"]:
        for d, L in cfg["rungs"]:
            for arch in ARCHITECTURES:
                if arch == "PhaseSum" and L != 1:
                    continue
                for seed in range(cfg["n_seeds"]):
                    plan.append((arch, n, d, L, seed))
    print(f"Total runs: {len(plan)}")

    t0 = time.time()
    rows = []
    for i, (arch, n, d, L, seed) in enumerate(plan, 1):
        spec = CyclicTaskSpec(n)
        m = build(arch, spec, d, L)
        params = count_params(m)
        print(f"\n[{i}/{len(plan)}] arch={arch}  n={n}  d={d}  L={L}  "
              f"seed={seed}  params={params:,}")
        rows.append(run_one(arch, spec, d, L, seed, cfg, device))
    dt = time.time() - t0
    print(f"\nTotal wallclock: {dt/60:.1f} min")

    agg = aggregate(rows)

    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump({
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in cfg.items()},
            "rows": rows,
            "agg":  {f"{a}|d{d}|L{L}|n{n}": v
                     for (a, d, L, n), v in agg.items()},
            "wallclock_sec": dt,
        }, f, indent=2, default=str)

    plot_capacity_curves(agg, cfg, os.path.join(args.outdir, "capacity_curves.png"))
    plot_closure_curves (agg, cfg, os.path.join(args.outdir, "closure_curves.png"))

    # --- Text summary ---
    lines = [f"Configuration: {args.config}",
             f"n values: {list(cfg['n_values'])}    "
             f"rungs (d, L): {[list(r) for r in cfg['rungs']]}",
             f"seeds: {cfg['n_seeds']}    "
             f"train depth <= {cfg['max_train_depth']}    "
             f"eval depth <= {cfg['max_eval_depth']}", ""]
    header = (f"{'n':>3s}  {'arch':>10s}  {'d':>4s}  {'L':>2s}  "
              f"{'params':>10s}  {'ID':>14s}  {'OOD':>14s}  "
              f"{'closure':>14s}")
    lines += [header, "-" * len(header)]
    for n in sorted(cfg["n_values"]):
        for arch in ARCHITECTURES:
            for (a, d, L, nn_), v in sorted(agg.items()):
                if a != arch or nn_ != n:
                    continue
                lines.append(
                    f"{n:>3d}  {arch:>10s}  {d:>4d}  {L:>2d}  "
                    f"{v['params']:>10,}  "
                    f"{v['id_mean']:.3f} ± {v['id_se']:.3f}  "
                    f"{v['ood_mean']:.3f} ± {v['ood_se']:.3f}  "
                    f"{v['closure_mean']:.3f} ± {v['closure_se']:.3f}")
        lines.append("")
    lines.append(f"Wallclock: {dt/60:.1f} min")
    summary = "\n".join(lines)
    print("\n" + "=" * 78); print(summary); print("=" * 78)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
