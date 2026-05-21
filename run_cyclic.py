"""
Experiment 4: scaling law over the cyclic group order n.

For each n in N_VALUES, train PhaseSumNet, RealAddNet, GatedComplexRNN,
GRUBaseline on the Z/n cyclic-rotation task and report:

  - ID accuracy
  - OOD accuracy (averaged over OOD depths)
  - closure under n TWIRLs (the group invariant)

Then plot all three as functions of n. The predicted pattern from
Experiments 1-3, restated:

  PhaseSumNet:      flat 1.0 across n (set-equivariance handles it)
  RealAddNet:       flat at chance (1/n) — provably can't fit Z/n linearly
  GatedComplexRNN:  ID stays high; OOD slopes downward as n grows
                    because angular resolution 2π/n shrinks while the
                    additive value path lets state magnitude drift
  GRUBaseline:      ID stays high; closure-under-n drops as n grows
                    because gating finds approximate n-state automata
                    that get worse at higher resolution

Run on H100:
    python3 run_cyclic.py --config full
Smoke:
    python3 run_cyclic.py --config smoke
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
from models_cyclic import (PhaseSumNet_n, RealAddNet_n, GatedComplexRNN_n,
                            GRUBaseline_n, count_params, matched_d_gru)
from train_cyclic import train_model
from analyze_cyclic import (probe_phase_sum_n, probe_real_add_n,
                             probe_closure_under_n_twirls)


CONFIGS = {
    "smoke": dict(
        n_values=(2, 3, 5),
        n_seeds=1, d_phase_sum=8, d_complex_rnn=12,
        max_train_depth=5, max_eval_depth=10,
        n_train=4_000, n_eval_per_depth=200, n_epochs=2,
        batch_size=128, lr=1e-2, eval_every_steps=30,
    ),
    "full": dict(
        n_values=(2, 3, 5, 7, 11, 13),
        n_seeds=3, d_phase_sum=16, d_complex_rnn=32,
        max_train_depth=5, max_eval_depth=20,
        n_train=60_000, n_eval_per_depth=2_000, n_epochs=15,
        batch_size=256, lr=5e-3, eval_every_steps=200,
    ),
}

OUTPUT_DIR  = "results_exp4"
MODEL_NAMES = ("PhaseSumNet", "RealAddNet", "GatedComplexRNN", "GRUBaseline")
COLORS  = {"PhaseSumNet":"C0", "RealAddNet":"C1",
            "GatedComplexRNN":"C2", "GRUBaseline":"C3"}
MARKERS = {"PhaseSumNet":"s",  "RealAddNet":"D",
            "GatedComplexRNN":"^", "GRUBaseline":"o"}


def build_models(spec, cfg):
    d_g = matched_d_gru(spec, cfg["d_complex_rnn"])
    return {
        "PhaseSumNet":     PhaseSumNet_n    (spec, d_model=cfg["d_phase_sum"]),
        "RealAddNet":      RealAddNet_n     (spec, d_model=cfg["d_phase_sum"]),
        "GatedComplexRNN": GatedComplexRNN_n(spec, d_model=cfg["d_complex_rnn"]),
        "GRUBaseline":     GRUBaseline_n    (spec, d_model=d_g),
    }


def run_one_n_one_seed(spec, cfg, seed, device):
    models = build_models(spec, cfg)
    params = {k: count_params(v) for k, v in models.items()}
    print(f"\n--- n={spec.n}, seed={seed} ---")
    for k, p in params.items():
        print(f"    {k:18s}  params={p:,}")

    common = dict(
        train_depth_range=(0, cfg["max_train_depth"]),
        eval_depths=tuple(range(0, cfg["max_eval_depth"] + 1)),
        n_train=cfg["n_train"], n_eval_per_depth=cfg["n_eval_per_depth"],
        batch_size=cfg["batch_size"], lr=cfg["lr"],
        n_epochs=cfg["n_epochs"], eval_every_steps=cfg["eval_every_steps"],
        device=device, seed=seed,
    )
    histories, trained = {}, {}
    for name, m in models.items():
        t, h = train_model(m, spec, tag=f"{name}-n{spec.n}-s{seed}", **common)
        trained[name], histories[name] = t, h

    probes = {
        "PhaseSumNet":     probe_phase_sum_n(trained["PhaseSumNet"], spec),
        "RealAddNet":      probe_real_add_n (trained["RealAddNet"],  spec),
    }
    for name in MODEL_NAMES:
        probes[f"{name}_closure"] = probe_closure_under_n_twirls(
            trained[name], spec, device)
    return {"n": spec.n, "seed": seed, "params": params,
            "histories": histories, "probes": probes}


def aggregate(results):
    """Group results by n and by model. Returns dict[n][model] = {accs, closures}."""
    by_n = {}
    for r in results:
        n = r["n"]
        by_n.setdefault(n, {"params": r["params"], "runs": []})
        by_n[n]["runs"].append(r)
    return by_n


def summarize(by_n, cfg):
    max_train = cfg["max_train_depth"]
    lines = ["Configuration: full",
             f"n values: {list(by_n)}    train depth ≤ {max_train}    "
             f"eval depth ≤ {cfg['max_eval_depth']}    seeds: {cfg['n_seeds']}",
             ""]
    header = f"{'n':>3s}  {'model':18s}  {'params':>7s}  " \
             f"{'ID acc':>14s}  {'OOD acc':>14s}  {'closure':>14s}"
    lines += [header, "-" * len(header)]
    for n, group in by_n.items():
        for name in MODEL_NAMES:
            in_acc, out_acc, closures = [], [], []
            for r in group["runs"]:
                fbd = r["histories"][name]["final_by_depth"]
                for d, a in fbd.items():
                    (in_acc if d <= max_train else out_acc).append(a)
                closures.append(
                    r["probes"][f"{name}_closure"]["frac_invariant_under_n_twirls"])
            def stat(v):
                a = np.array(v)
                return (f"{a.mean():.3f} ± {a.std()/math.sqrt(len(a)):.3f}"
                        if len(a) else "n/a")
            params = group["params"][name]
            lines.append(f"{n:>3d}  {name:18s}  {params:>7,}  "
                         f"{stat(in_acc):>14s}  {stat(out_acc):>14s}  "
                         f"{stat(closures):>14s}")
        lines.append("")
    return "\n".join(lines)


def plot_scaling(by_n, cfg, path):
    """Three panels: ID acc, OOD acc, closure — all as functions of n."""
    ns = sorted(by_n)
    max_train = cfg["max_train_depth"]

    def gather(metric):
        # metric ∈ {"ID","OOD","closure"}
        means, stderrs = {name: [] for name in MODEL_NAMES}, \
                          {name: [] for name in MODEL_NAMES}
        for n in ns:
            for name in MODEL_NAMES:
                vals = []
                for r in by_n[n]["runs"]:
                    if metric == "closure":
                        vals.append(r["probes"][f"{name}_closure"]
                                     ["frac_invariant_under_n_twirls"])
                    else:
                        fbd = r["histories"][name]["final_by_depth"]
                        sub = [a for d, a in fbd.items()
                                 if (d <= max_train) == (metric == "ID")]
                        vals.append(np.mean(sub))
                a = np.array(vals)
                means[name].append(a.mean())
                stderrs[name].append(a.std() / math.sqrt(len(a)))
        return means, stderrs

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, metric, title in zip(
        axes, ("ID", "OOD", "closure"),
        ("In-distribution accuracy",
         "Out-of-distribution accuracy",
         "Closure under n·TWIRL")):
        means, stderrs = gather(metric)
        for name in MODEL_NAMES:
            m = np.array(means[name]); s = np.array(stderrs[name])
            ax.fill_between(ns, m - s, m + s, alpha=0.18, color=COLORS[name])
            ax.plot(ns, m, marker=MARKERS[name], color=COLORS[name], lw=2,
                    label=name)
        # Chance line (1/n) for the accuracy panels.
        if metric in ("ID", "OOD"):
            ax.plot(ns, [1.0 / n for n in ns],
                    color="k", lw=0.7, ls=":", label="chance (1/n)")
        ax.set_xticks(ns)
        ax.set_xlabel("Group order n")
        ax.set_title(title)
        ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("Accuracy / invariance fraction")
    axes[2].legend(loc="lower left", fontsize=8)
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
    print(json.dumps({k: v for k, v in cfg.items()}, indent=2,
                     default=str))
    print(f"Device: {device}")
    print(f"Total runs: {len(cfg['n_values'])} × {cfg['n_seeds']} × 4 = "
          f"{len(cfg['n_values']) * cfg['n_seeds'] * 4}")

    t0 = time.time()
    all_results = []
    for n in cfg["n_values"]:
        spec = CyclicTaskSpec(n)
        for seed in range(cfg["n_seeds"]):
            all_results.append(run_one_n_one_seed(spec, cfg, seed=seed,
                                                    device=device))
    dt = time.time() - t0
    print(f"\nTotal wallclock: {dt/60:.1f} min")

    by_n = aggregate(all_results)

    slim = []
    for r in all_results:
        slim.append({
            "n": r["n"], "seed": r["seed"], "params": r["params"],
            "final_acc":      {k: r["histories"][k]["final_acc"]
                                for k in r["histories"]},
            "final_by_depth": {k: r["histories"][k]["final_by_depth"]
                                for k in r["histories"]},
            "probes": r["probes"],
        })
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump({"config": {k: v for k, v in cfg.items()},
                   "results": slim, "wallclock_sec": dt}, f, indent=2,
                  default=str)

    plot_scaling(by_n, cfg, os.path.join(args.outdir, "scaling_law.png"))

    summary = summarize(by_n, cfg)
    print("\n" + "=" * 70); print(summary); print("=" * 70)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(summary + f"\nWallclock: {dt/60:.1f} min\n")


if __name__ == "__main__":
    main()
