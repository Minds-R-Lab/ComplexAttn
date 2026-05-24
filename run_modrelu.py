"""Multi-seed test of ModReLUStackedRMC vs DeepMLP at depths 1, 2, 4 on
the dynamical-system task. Saves incrementally."""

import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dynamical_data import make_loaders
from modrelu_rmc import ModReLUStackedRMC, count_trainable
from stacked import DeepMLP
from train import train_model

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "modrelu.json"


def load_state():
    if OUT.exists(): return json.loads(OUT.read_text())
    return {"runs": []}


def save_state(s): OUT.write_text(json.dumps(s, indent=2))
def done(state, L, seed, name):
    return any(r["L"] == L and r["seed"] == seed and r["model"] == name for r in state["runs"])


def run_one(L, seed, name, epochs=10):
    tr, va = make_loaders(n_train=3000, n_val=500, batch_size=64, seed=seed)
    torch.manual_seed(seed)
    if name == "ModReLU-RMC":
        m = ModReLUStackedRMC(num_layers=L); lr = 3e-3
    else:
        m = DeepMLP(num_layers=L); lr = 2e-3
    h = train_model(m, tr, va, epochs=epochs, lr=lr, name=f"{name}-L{L}-s{seed}",
                    save_to=None, log_every=0, grad_clip=1.0)
    n_eff = sum(p.numel() for p in m.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    return float(max(h["val_acc"])), count_trainable(m), int(n_eff)


def main():
    torch.set_num_threads(1)
    Ls = [1, 2, 4]; seeds = [0, 1, 2]
    state = load_state()
    print(f"resuming with {len(state['runs'])} done")
    for L in Ls:
        for seed in seeds:
            for name in ["ModReLU-RMC", "MLP"]:
                if done(state, L, seed, name): continue
                acc, npar, neff = run_one(L, seed, name)
                state["runs"].append({"L": L, "seed": seed, "model": name,
                                       "best_val_acc": acc, "n_params": npar, "n_effective": neff})
                save_state(state)
                print(f"  L={L} seed={seed} {name:12s} best={acc:.4f} (params={npar}, effective={neff})")
    # aggregate
    print(f"\n=== ModReLU-RMC vs DeepMLP, n=3000, 10 epochs ===")
    print(f"{'L':>3s}  {'RMC mean':>10s}  {'RMC std':>9s}  {'MLP mean':>10s}  {'MLP std':>9s}  {'gap pp':>8s}  RMC_params  RMC_eff  MLP_params")
    summary = {"L": Ls, "rmc_mean": [], "rmc_std": [], "mlp_mean": [], "mlp_std": [],
               "rmc_params": [], "rmc_eff": [], "mlp_params": []}
    for L in Ls:
        rmc = np.array([r["best_val_acc"] for r in state["runs"] if r["L"] == L and r["model"] == "ModReLU-RMC"])
        mlp = np.array([r["best_val_acc"] for r in state["runs"] if r["L"] == L and r["model"] == "MLP"])
        ns = min(len(rmc), len(mlp))
        if ns == 0: continue
        rmc, mlp = rmc[:ns], mlp[:ns]
        rmu = rmc.mean(); rsd = rmc.std(ddof=1) if ns > 1 else 0.0
        mmu = mlp.mean(); msd = mlp.std(ddof=1) if ns > 1 else 0.0
        rp = next(r["n_params"] for r in state["runs"] if r["L"] == L and r["model"] == "ModReLU-RMC")
        rpe = next(r["n_effective"] for r in state["runs"] if r["L"] == L and r["model"] == "ModReLU-RMC")
        mp = next(r["n_params"] for r in state["runs"] if r["L"] == L and r["model"] == "MLP")
        summary["rmc_mean"].append(float(rmu)); summary["rmc_std"].append(float(rsd))
        summary["mlp_mean"].append(float(mmu)); summary["mlp_std"].append(float(msd))
        summary["rmc_params"].append(rp); summary["rmc_eff"].append(rpe); summary["mlp_params"].append(mp)
        print(f"{L:>3d}  {rmu:>10.4f}  {rsd:>9.4f}  {mmu:>10.4f}  {msd:>9.4f}  {(rmu-mmu)*100:>+8.2f}  {rp:>10d}  {rpe:>7d}  {mp:>10d}")
    state["aggregated"] = summary; save_state(state)

    # plot
    Ls_done = summary["L"][:len(summary["rmc_mean"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Ls_done, summary["rmc_mean"], yerr=summary["rmc_std"], fmt="o-",
                color="C0", label="ModReLU-RMC", linewidth=2, markersize=10, capsize=6)
    ax.errorbar(Ls_done, summary["mlp_mean"], yerr=summary["mlp_std"], fmt="o-",
                color="C7", label="DeepMLP", linewidth=2, markersize=10, capsize=6)
    for i, L in enumerate(Ls_done):
        gap = (summary["rmc_mean"][i] - summary["mlp_mean"][i]) * 100
        peak = max(summary["rmc_mean"][i] + summary["rmc_std"][i],
                   summary["mlp_mean"][i] + summary["mlp_std"][i])
        ax.annotate(f"{gap:+.1f}pp", (L, peak + 0.015), fontsize=9, color="C3", ha="center", weight="bold")
    ax.set_xlabel("number of layers L"); ax.set_ylabel("best val accuracy (n=3000)")
    ax.set_xticks(Ls_done)
    ax.set_title("ModReLU-stacked RMC vs DeepMLP\n"
                 "(phase-equivariant nonlinearity between cells)")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "modrelu_depth.png", dpi=130); plt.close(fig)
    print(f"\nsaved {RESULTS / 'modrelu_depth.png'}")


if __name__ == "__main__":
    main()
