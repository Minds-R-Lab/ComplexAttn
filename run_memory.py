"""Multi-seed depth experiment for MemoryNet vs DeepMLP."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from dynamical_data import make_loaders
from memory_net import MemoryNet, count_trainable
from stacked import DeepMLP
from train import train_model

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "memory.json"

def load(): return json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}
def save(s): OUT.write_text(json.dumps(s, indent=2))
def done(state, L, seed, name):
    return any(r["L"]==L and r["seed"]==seed and r["model"]==name for r in state["runs"])

def run_one(L, seed, name, epochs=10):
    tr, va = make_loaders(n_train=3000, n_val=500, batch_size=64, seed=seed)
    torch.manual_seed(seed)
    m = MemoryNet(num_layers=L) if name == "Memory" else DeepMLP(num_layers=L)
    h = train_model(m, tr, va, epochs=epochs, lr=2e-3, name=f"{name}-L{L}-s{seed}",
                    save_to=None, log_every=0, grad_clip=1.0)
    return float(max(h["val_acc"])), count_trainable(m)

def main():
    torch.set_num_threads(1)
    Ls = [1, 2, 4]; seeds = [0, 1, 2]
    state = load()
    print(f"resuming with {len(state['runs'])} done")
    for L in Ls:
        for seed in seeds:
            for name in ["Memory", "MLP"]:
                if done(state, L, seed, name): continue
                acc, npar = run_one(L, seed, name)
                state["runs"].append({"L": L, "seed": seed, "model": name,
                                       "best_val_acc": acc, "n_params": npar})
                save(state)
                print(f"  L={L} seed={seed} {name:8s} best={acc:.4f} (params={npar})")
    print(f"\n=== MemoryNet vs DeepMLP, n=3000, 10 epochs, 3 seeds ===")
    print(f"{'L':>3s}  {'Mem mean':>10s}  {'Mem std':>9s}  {'MLP mean':>10s}  {'MLP std':>9s}  {'gap pp':>8s}  Mem_p   MLP_p   M_wins")
    summary = {"L": Ls, "m_mean": [], "m_std": [], "mlp_mean": [], "mlp_std": [],
               "m_params": [], "mlp_params": [], "m_wins": []}
    for L in Ls:
        mem = np.array([r["best_val_acc"] for r in state["runs"] if r["L"]==L and r["model"]=="Memory"])
        mlp = np.array([r["best_val_acc"] for r in state["runs"] if r["L"]==L and r["model"]=="MLP"])
        ns = min(len(mem), len(mlp))
        if ns == 0: continue
        mem, mlp = mem[:ns], mlp[:ns]
        mmu = mem.mean(); msd = mem.std(ddof=1) if ns>1 else 0.0
        pmu = mlp.mean(); psd = mlp.std(ddof=1) if ns>1 else 0.0
        gap = (mmu-pmu)*100; wins = int((mem>mlp).sum())
        mp = next(r["n_params"] for r in state["runs"] if r["L"]==L and r["model"]=="Memory")
        pp = next(r["n_params"] for r in state["runs"] if r["L"]==L and r["model"]=="MLP")
        summary["m_mean"].append(float(mmu)); summary["m_std"].append(float(msd))
        summary["mlp_mean"].append(float(pmu)); summary["mlp_std"].append(float(psd))
        summary["m_params"].append(mp); summary["mlp_params"].append(pp); summary["m_wins"].append(wins)
        print(f"{L:>3d}  {mmu:>10.4f}  {msd:>9.4f}  {pmu:>10.4f}  {psd:>9.4f}  {gap:>+8.2f}  {mp:>5d}   {pp:>5d}   {wins}/{ns}")
    state["aggregated"] = summary; save(state)
    Ls_done = summary["L"][:len(summary["m_mean"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Ls_done, summary["m_mean"], yerr=summary["m_std"], fmt="o-",
                color="C1", label="MemoryNet", linewidth=2, markersize=10, capsize=6)
    ax.errorbar(Ls_done, summary["mlp_mean"], yerr=summary["mlp_std"], fmt="o-",
                color="C7", label="DeepMLP-32", linewidth=2, markersize=10, capsize=6)
    for i, L in enumerate(Ls_done):
        gap = (summary["m_mean"][i] - summary["mlp_mean"][i]) * 100
        peak = max(summary["m_mean"][i] + summary["m_std"][i],
                   summary["mlp_mean"][i] + summary["mlp_std"][i])
        ax.annotate(f"{gap:+.1f}pp", (L, peak + 0.015), fontsize=9,
                    color="C2" if gap >= 0 else "C3", ha="center", weight="bold")
    ax.set_xlabel("number of layers L"); ax.set_ylabel("best val accuracy (n=3000)")
    ax.set_xticks(Ls_done)
    ax.set_title("MemoryNet (retrieval) vs DeepMLP on dynamical task")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "memory_depth.png", dpi=130); plt.close(fig)
    print(f"\nsaved {RESULTS / 'memory_depth.png'}")

if __name__ == "__main__":
    main()
