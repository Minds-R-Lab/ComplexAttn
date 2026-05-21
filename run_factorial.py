"""
Experiment 6: factorial ablation across n.

Runs the 6 factorial-cell architectures from models_factorial.py on
Z/n for n in {2, 3, 5, 7, 11}, 3 seeds each. Total: 6 * 5 * 3 = 90 runs.

Outputs results_exp6/{summary.txt, results.json, heatmap.png}.

The headline figure (heatmap) shows OOD accuracy in a 2x2 grid of
factorial cells, one panel per n, with PhaseSumNet (= ComplexMulPer)
in the bottom-right of each. The eye-test for the paper's central
claim: if the PERIODIC-readout column wins regardless of composition,
the paper's framing is wrong; if only ComplexMulPer wins, the paper's
framing is right.
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
from models_factorial import ARCH_FACTORY, make_arch, count_params
from train_cyclic import train_model


CONFIGS = {
    "smoke": dict(
        n_values=(3, 5),
        archs=("RealAddLin", "RealAddPer", "ComplexMulPer"),
        d=8,
        n_seeds=1,
        max_train_depth=5, max_eval_depth=10,
        n_train=4_000, n_eval_per_depth=200,
        n_epochs=2, batch_size=128, lr=1e-2,
        eval_every_steps=30,
    ),
    "full": dict(
        n_values=(2, 3, 5, 7, 11),
        archs=("RealAddLin", "RealAddPer", "ComplexMulLin",
                "ComplexMulPer", "GRULin", "GRUPer"),
        d=16,
        n_seeds=3,
        max_train_depth=5, max_eval_depth=20,
        n_train=60_000, n_eval_per_depth=2_000,
        n_epochs=20, batch_size=256, lr=5e-3,
        eval_every_steps=200,
    ),
}

OUTPUT_DIR = "results_exp6"


def run_one(arch_name, spec, d, seed, cfg, device):
    model = make_arch(arch_name, spec, d=d)
    tag = f"{arch_name}-n{spec.n}-s{seed}"
    print(f"\n[{tag}]  params={count_params(model):,}")
    trained, hist = train_model(
        model, spec,
        train_depth_range=(0, cfg["max_train_depth"]),
        eval_depths=tuple(range(0, cfg["max_eval_depth"] + 1)),
        n_train=cfg["n_train"], n_eval_per_depth=cfg["n_eval_per_depth"],
        batch_size=cfg["batch_size"], lr=cfg["lr"],
        n_epochs=cfg["n_epochs"], eval_every_steps=cfg["eval_every_steps"],
        device=device, seed=seed, tag=tag)

    fbd = hist["final_by_depth"]
    ind = [a for d_, a in fbd.items() if d_ <= cfg["max_train_depth"]]
    ood = [a for d_, a in fbd.items() if d_ >  cfg["max_train_depth"]]
    return {
        "arch": arch_name, "n": spec.n, "seed": seed,
        "params": count_params(model),
        "id_acc":  float(np.mean(ind)),
        "ood_acc": float(np.mean(ood)),
        "best_ood": hist["best_ood"],
        "final_by_depth": fbd,
        "wallclock_sec": hist["wallclock_sec"],
    }


def aggregate(rows):
    by = {}
    for r in rows:
        key = (r["arch"], r["n"])
        by.setdefault(key, []).append(r)
    out = {}
    for (arch, n), lst in by.items():
        ids  = np.array([x["id_acc"]  for x in lst])
        oods = np.array([x["ood_acc"] for x in lst])
        out[(arch, n)] = {
            "params":   lst[0]["params"],
            "id_mean":  float(ids.mean()),
            "id_se":    float(ids.std() / math.sqrt(len(ids))),
            "ood_mean": float(oods.mean()),
            "ood_se":   float(oods.std() / math.sqrt(len(oods))),
            "n_seeds":  len(lst),
        }
    return out


def plot_heatmap(agg, cfg, path):
    """Per n, a 2x2 grid showing OOD accuracy in the additive/multiplicative
    by linear/periodic factorial. Plus GRU comparison as a side panel.
    """
    ns = sorted(cfg["n_values"])
    grid_rows = ["Additive\ncomposition", "Multiplicative\ncomposition"]
    grid_cols = ["Linear readout", "Periodic readout"]
    cell_arch = [["RealAddLin",    "RealAddPer"],
                 ["ComplexMulLin", "ComplexMulPer"]]

    n_panels = len(ns) + 1  # one panel per n + one summary panel
    fig, axes = plt.subplots(1, len(ns), figsize=(3.6 * len(ns), 4.2),
                              squeeze=False)
    for ax, n in zip(axes[0], ns):
        mat = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                arch = cell_arch[i][j]
                key = (arch, n)
                if key in agg:
                    mat[i, j] = agg[key]["ood_mean"]
                else:
                    mat[i, j] = float("nan")
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(grid_cols, fontsize=8)
        ax.set_yticklabels(grid_rows, fontsize=8)
        for i in range(2):
            for j in range(2):
                txt = f"{mat[i,j]:.2f}"
                color = "black" if 0.3 < mat[i, j] < 0.85 else "white"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")
        chance = 1.0 / n
        ax.set_title(f"$\\mathbb{{Z}}/{n}$  (chance = {chance:.2f})")
    fig.suptitle("Factorial ablation: OOD accuracy across composition × readout")
    plt.tight_layout()
    plt.savefig(path, dpi=140); plt.close()


def plot_gru_comparison(agg, cfg, path):
    """OOD vs n for GRULin and GRUPer side by side, with PhaseSumNet
    reference line.  Skipped if the relevant archs weren't run."""
    needed = {"ComplexMulPer", "GRULin", "GRUPer", "RealAddPer"}
    if not needed.issubset(set(cfg["archs"])):
        print(f"  [plot_gru_comparison] skipping: needs {needed - set(cfg['archs'])}")
        return
    ns = sorted(cfg["n_values"])
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for arch, color, marker in [
        ("ComplexMulPer", "C0", "s"),
        ("GRULin",        "C3", "o"),
        ("GRUPer",        "C4", "v"),
        ("RealAddPer",    "C2", "D"),
    ]:
        oods = [agg[(arch, n)]["ood_mean"] for n in ns]
        ses  = [agg[(arch, n)]["ood_se"]   for n in ns]
        ax.fill_between(ns, [o - s for o, s in zip(oods, ses)],
                            [o + s for o, s in zip(oods, ses)],
                        color=color, alpha=0.18)
        ax.plot(ns, oods, marker=marker, color=color, lw=2,
                label=arch)
    ax.plot(ns, [1.0 / n for n in ns], color="k", lw=0.7, ls=":",
            label="chance (1/n)")
    ax.set_xticks(ns); ax.set_xlabel("Group order n")
    ax.set_ylabel("OOD accuracy")
    ax.set_title("Does periodic readout rescue the GRU on Z/n?")
    ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


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

    plan = [(arch, n, seed)
            for arch in cfg["archs"]
            for n in cfg["n_values"]
            for seed in range(cfg["n_seeds"])]
    print(f"Total runs: {len(plan)}")

    t0 = time.time()
    rows = []
    for i, (arch, n, seed) in enumerate(plan, 1):
        spec = CyclicTaskSpec(n)
        print(f"\n[{i}/{len(plan)}] arch={arch}  n={n}  seed={seed}")
        rows.append(run_one(arch, spec, cfg["d"], seed, cfg, device))
    dt = time.time() - t0
    print(f"\nTotal wallclock: {dt/60:.1f} min")

    agg = aggregate(rows)
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump({
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in cfg.items()},
            "rows": rows,
            "agg":  {f"{a}|n{n}": v for (a, n), v in agg.items()},
            "wallclock_sec": dt,
        }, f, indent=2, default=str)

    plot_heatmap        (agg, cfg, os.path.join(args.outdir, "factorial_heatmap.png"))
    plot_gru_comparison (agg, cfg, os.path.join(args.outdir, "gru_periodic_rescue.png"))

    # Text summary -- the headline table
    lines = [f"Configuration: {args.config}",
             f"n values: {list(cfg['n_values'])}    "
             f"archs: {list(cfg['archs'])}    "
             f"seeds: {cfg['n_seeds']}    d={cfg['d']}",
             f"Train depth <= {cfg['max_train_depth']}    "
             f"Eval depth <= {cfg['max_eval_depth']}    "
             f"n_train = {cfg['n_train']}",
             ""]
    # Header
    header = f"{'arch':>16s}  {'n':>3s}  {'params':>7s}  {'ID acc':>14s}  {'OOD acc':>14s}  {'OOD - chance':>14s}"
    lines += [header, "-" * len(header)]
    for arch in cfg["archs"]:
        for n in sorted(cfg["n_values"]):
            v = agg.get((arch, n))
            if v is None: continue
            chance = 1.0 / n
            lines.append(
                f"{arch:>16s}  {n:>3d}  {v['params']:>7,}  "
                f"{v['id_mean']:.3f} ± {v['id_se']:.3f}  "
                f"{v['ood_mean']:.3f} ± {v['ood_se']:.3f}  "
                f"{(v['ood_mean'] - chance):+.3f}")
        lines.append("")
    lines += ["", f"Wallclock: {dt/60:.1f} min"]
    summary = "\n".join(lines)
    print("\n" + "=" * 80); print(summary); print("=" * 80)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
