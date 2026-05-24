"""Multi-seed depth experiment for TropicalNet vs DeepMLP on the dynamical
task at n=3000. Same protocol as run_stacked.py, saves incrementally."""

import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dynamical_data import make_loaders
from stacked import DeepMLP
from train import train_model
from tropical_net import TropicalNet, count_trainable

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "tropical.json"


def load(): return json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}
def save(s): OUT.write_text(json.dumps(s, indent=2))
def done(state, L, seed, name):
    return any(r["L"]==L and r["seed"]==seed and r["model"]==name for r in state["runs"])


def run_one(L, seed, name, epochs=10):
    tr, va = make_loaders(n_train=3000, n_val=500, batch_size=64, seed=seed)
    torch.manual_seed(seed)
    if name == "Tropical":
        m = TropicalNet(num_layers=L); lr = 2e-3
    else:
        m = DeepMLP(num_layers=L); lr = 2e-3
    h = train_model(m, tr, va, epochs=epochs, lr=lr, name=f"{name}-L{L}-s{seed}",
                    save_to=None, log_every=0, grad_clip=1.0)
    return float(max(h["val_acc"])), count_trainable(m)


def main():
    torch.set_num_threads(1)
    Ls = [1, 2, 4]; seeds = [0, 1, 2]
    state = load()
    print(f"resuming with {len(state['runs'])} done")
    for L in Ls:
        for seed in seeds:
            for name in ["Tropical", "MLP"]:
                if done(state, L, seed, name): continue
                acc, npar = run_one(L, seed, name)
                state["runs"].append({"L": L, "seed": seed, "model": name,
                                       "best_val_acc": acc, "n_params": npar})
                save(state)
                print(f"  L={L} seed={seed} {name:8s} best={acc:.4f} (params={npar})")
    # aggregate
    print(f"\n=== Tropical vs DeepMLP, n=3000, 10 epochs, 3 seeds ===")
    print(f"{'L':>3s}  {'Trop mean':>10s}  {'Trop std':>9s}  {'MLP mean':>10s}  {'MLP std':>9s}  {'gap pp':>8s}  Trop_p  MLP_p  T_wins")
    summary = {"L": Ls, "t_mean": [], "t_std": [], "m_mean": [], "m_std": [],
               "t_params": [], "m_params": [], "t_wins": []}
    for L in Ls:
        t = np.array([r["best_val_acc"] for r in state["runs"] if r["L"]==L and r["model"]=="Tropical"])
        m = np.array([r["best_val_acc"] for r in state["runs"] if r["L"]==L and r["model"]=="MLP"])
        ns = min(len(t), len(m))
        if ns == 0: continue
        t, m = t[:ns], m[:ns]
        tmu, tsd = t.mean(), t.std(ddof=1) if ns>1 else 0.0
        mmu, msd = m.mean(), m.std(ddof=1) if ns>1 else 0.0
        gap = (tmu-mmu)*100; wins = int((t>m).sum())
        tp = next(r["n_params"] for r in state["runs"] if r["L"]==L and r["model"]=="Tropical")
        mp = next(r["n_params"] for r in state["runs"] if r["L"]==L and r["model"]=="MLP")
        summary["t_mean"].append(float(tmu)); summary["t_std"].append(float(tsd))
        summary["m_mean"].append(float(mmu)); summary["m_std"].append(float(msd))
        summary["t_params"].append(tp); summary["m_params"].append(mp); summary["t_wins"].append(wins)
        print(f"{L:>3d}  {tmu:>10.4f}  {tsd:>9.4f}  {mmu:>10.4f}  {msd:>9.4f}  {gap:>+8.2f}  {tp:>6d}  {mp:>5d}  {wins}/{ns}")
    state["aggregated"] = summary; save(state)

    Ls_done = summary["L"][:len(summary["t_mean"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Ls_done, summary["t_mean"], yerr=summary["t_std"], fmt="o-",
                color="C2", label="TropicalNet", linewidth=2, markersize=10, capsize=6)
    ax.errorbar(Ls_done, summary["m_mean"], yerr=summary["m_std"], fmt="o-",
                color="C7", label="DeepMLP-32", linewidth=2, markersize=10, capsize=6)
    for i, L in enumerate(Ls_done):
        gap = (summary["t_mean"][i] - summary["m_mean"][i]) * 100
        peak = max(summary["t_mean"][i] + summary["t_std"][i],
                   summary["m_mean"][i] + summary["m_std"][i])
        ax.annotate(f"{gap:+.1f}pp", (L, peak + 0.015), fontsize=9,
                    color="C3" if gap < 0 else "C2", ha="center", weight="bold")
    ax.set_xlabel("number of hidden layers L")
    ax.set_ylabel("best val accuracy (n=3000)")
    ax.set_xticks(Ls_done)
    ax.set_title("TropicalNet (max,+) vs DeepMLP (+,x) on dynamical task")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "tropical_depth.png", dpi=130); plt.close(fig)
    print(f"\nsaved {RESULTS / 'tropical_depth.png'}")


if __name__ == "__main__":
    main()
