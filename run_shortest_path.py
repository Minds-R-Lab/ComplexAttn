"""Primitive-task match experiment: shortest-path task where MinPlus
(Bellman-Ford as primitive) should beat MLP. Multi-seed, multi-depth."""

import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shortest_path_data import make_loaders
from stacked import DeepMLP
from train import train_model
from tropical_net import MinPlusNet, count_trainable

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "shortest_path.json"

def load(): return json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}
def save(s): OUT.write_text(json.dumps(s, indent=2))
def done(state, L, seed, name):
    return any(r["L"]==L and r["seed"]==seed and r["model"]==name for r in state["runs"])


def run_one(L, seed, name, epochs=10):
    tr, va = make_loaders(n_train=3000, n_val=500, batch_size=64, seed=seed)
    torch.manual_seed(seed)
    if name == "MinPlus":
        m = MinPlusNet(num_layers=L, hidden_dim=32)   # exact param match w/ MLP
        lr = 2e-3
    else:
        m = DeepMLP(num_layers=L)
        lr = 2e-3
    h = train_model(m, tr, va, epochs=epochs, lr=lr, name=f"{name}-L{L}-s{seed}",
                    save_to=None, log_every=0, grad_clip=1.0)
    return float(max(h["val_acc"])), count_trainable(m)


def main():
    torch.set_num_threads(1)
    Ls = [2, 4]   # min-plus needs >=N=8 layers for exact BF; here we test 2 and 4
    seeds = [0, 1, 2]
    state = load()
    print(f"resuming with {len(state['runs'])} done")
    for L in Ls:
        for seed in seeds:
            for name in ["MinPlus", "MLP"]:
                if done(state, L, seed, name): continue
                acc, npar = run_one(L, seed, name)
                state["runs"].append({"L": L, "seed": seed, "model": name,
                                       "best_val_acc": acc, "n_params": npar})
                save(state)
                print(f"  L={L} seed={seed} {name:8s} best={acc:.4f} (params={npar})")
    # aggregate
    print(f"\n=== Shortest-path task (3 seeds, 10 epochs) ===")
    print(f"{'L':>3s}  {'MinPlus mean':>12s}  {'MinPlus std':>11s}  {'MLP mean':>10s}  {'MLP std':>9s}  {'gap pp':>8s}  M_wins  params")
    summary = {"L": Ls, "mp_mean": [], "mp_std": [], "mlp_mean": [], "mlp_std": [],
               "mp_wins": [], "params": []}
    for L in Ls:
        mp = np.array([r["best_val_acc"] for r in state["runs"] if r["L"]==L and r["model"]=="MinPlus"])
        mlp = np.array([r["best_val_acc"] for r in state["runs"] if r["L"]==L and r["model"]=="MLP"])
        ns = min(len(mp), len(mlp))
        if ns == 0: continue
        mp, mlp = mp[:ns], mlp[:ns]
        mpu, mps = mp.mean(), mp.std(ddof=1) if ns>1 else 0.0
        plu, pls = mlp.mean(), mlp.std(ddof=1) if ns>1 else 0.0
        gap = (mpu-plu)*100; wins = int((mp>mlp).sum())
        pcount = next(r["n_params"] for r in state["runs"] if r["L"]==L and r["model"]=="MinPlus")
        summary["mp_mean"].append(float(mpu)); summary["mp_std"].append(float(mps))
        summary["mlp_mean"].append(float(plu)); summary["mlp_std"].append(float(pls))
        summary["mp_wins"].append(wins); summary["params"].append(pcount)
        print(f"{L:>3d}  {mpu:>12.4f}  {mps:>11.4f}  {plu:>10.4f}  {pls:>9.4f}  {gap:>+8.2f}  {wins}/{ns}    {pcount}")
    state["aggregated"] = summary; save(state)
    # plot
    Ls_done = summary["L"][:len(summary["mp_mean"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Ls_done, summary["mp_mean"], yerr=summary["mp_std"], fmt="o-",
                color="C2", label="MinPlusNet  (Bellman-Ford primitive)",
                linewidth=2, markersize=11, capsize=6)
    ax.errorbar(Ls_done, summary["mlp_mean"], yerr=summary["mlp_std"], fmt="o-",
                color="C7", label="DeepMLP-32",
                linewidth=2, markersize=11, capsize=6)
    for i, L in enumerate(Ls_done):
        gap = (summary["mp_mean"][i] - summary["mlp_mean"][i]) * 100
        peak = max(summary["mp_mean"][i] + summary["mp_std"][i],
                   summary["mlp_mean"][i] + summary["mlp_std"][i])
        ax.annotate(f"{gap:+.1f}pp", (L, peak + 0.015), fontsize=10,
                    color="C2" if gap >= 0 else "C3", ha="center", weight="bold")
    ax.set_xlabel("number of hidden layers L")
    ax.set_ylabel("best val accuracy (n=3000)")
    ax.set_xticks(Ls_done)
    ax.set_title("Shortest-path-through-noisy-graph task\n"
                 "MinPlusNet (matched primitive) vs DeepMLP — same param count")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "shortest_path.png", dpi=130); plt.close(fig)
    print(f"\nsaved {RESULTS / 'shortest_path.png'}")


if __name__ == "__main__":
    main()
