"""Multi-seed depth comparison of MRBNet vs DeepMLP on dynamical task."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dynamical_data import make_loaders
from mrb_net import MRBNet, count_trainable
from stacked import DeepMLP
from train import train_model

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "mrb.json"

def load(): return json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}
def save(s): OUT.write_text(json.dumps(s, indent=2))
def done(state, L, seed, name):
    return any(r["L"]==L and r["seed"]==seed and r["model"]==name for r in state["runs"])

def run_one(L, seed, name, epochs=10):
    tr, va = make_loaders(n_train=3000, n_val=500, batch_size=64, seed=seed)
    torch.manual_seed(seed)
    m = MRBNet(num_layers=L) if name == "MRB" else DeepMLP(num_layers=L)
    h = train_model(m, tr, va, epochs=epochs, lr=2e-3, name=f"{name}-L{L}-s{seed}",
                    save_to=None, log_every=0, grad_clip=1.0)
    return float(max(h["val_acc"])), count_trainable(m)

def main():
    torch.set_num_threads(1)
    Ls = [1, 2, 4]; seeds = [0, 1, 2]
    state = load()
    for L in Ls:
        for seed in seeds:
            for name in ["MRB", "MLP"]:
                if done(state, L, seed, name): continue
                acc, npar = run_one(L, seed, name)
                state["runs"].append({"L": L, "seed": seed, "model": name,
                                       "best_val_acc": acc, "n_params": npar})
                save(state)
                print(f"  L={L} seed={seed} {name:4s} best={acc:.4f} (params={npar})")
    print(f"\n=== MRBNet vs DeepMLP, n=3000, 10 epochs, 3 seeds ===")
    print(f"{'L':>3s}  {'MRB mean':>10s}  {'MRB std':>9s}  {'MLP mean':>10s}  {'MLP std':>9s}  {'gap pp':>8s}  MRB_wins  params(M/MLP)")
    summary = {"L": Ls, "mrb_mean": [], "mrb_std": [], "mlp_mean": [], "mlp_std": [],
               "mrb_wins": [], "mrb_params": [], "mlp_params": []}
    for L in Ls:
        mrb = np.array([r["best_val_acc"] for r in state["runs"] if r["L"]==L and r["model"]=="MRB"])
        mlp = np.array([r["best_val_acc"] for r in state["runs"] if r["L"]==L and r["model"]=="MLP"])
        ns = min(len(mrb), len(mlp))
        if ns == 0: continue
        mrb, mlp = mrb[:ns], mlp[:ns]
        rmu = mrb.mean(); rsd = mrb.std(ddof=1) if ns>1 else 0.0
        mmu = mlp.mean(); msd = mlp.std(ddof=1) if ns>1 else 0.0
        gap = (rmu-mmu)*100; wins = int((mrb>mlp).sum())
        rp = next(r["n_params"] for r in state["runs"] if r["L"]==L and r["model"]=="MRB")
        mp = next(r["n_params"] for r in state["runs"] if r["L"]==L and r["model"]=="MLP")
        summary["mrb_mean"].append(float(rmu)); summary["mrb_std"].append(float(rsd))
        summary["mlp_mean"].append(float(mmu)); summary["mlp_std"].append(float(msd))
        summary["mrb_wins"].append(wins)
        summary["mrb_params"].append(rp); summary["mlp_params"].append(mp)
        print(f"{L:>3d}  {rmu:>10.4f}  {rsd:>9.4f}  {mmu:>10.4f}  {msd:>9.4f}  {gap:>+8.2f}  {wins}/{ns}      {rp}/{mp}")
    state["aggregated"] = summary; save(state)

if __name__ == "__main__":
    main()
