"""Round 3: 4 new blocks vs MLP, dynamical L=2, 3 seeds × 10 epochs."""
import json
from pathlib import Path
import numpy as np
import torch
from dynamical_data import make_loaders
from novel_blocks3 import pairwise_net, tropmax_net, tropmin_net, anchor_net, count_p
from stacked import DeepMLP
from train import train_model

OUT = Path("results/novel3.json")
def load(): return json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}
def save(s): OUT.write_text(json.dumps(s, indent=2))
def done(state, seed, name):
    return any(r["seed"]==seed and r["model"]==name for r in state["runs"])

torch.set_num_threads(1)
configs = [
    ("PairwiseDiff", pairwise_net),
    ("TropMax",      tropmax_net),
    ("TropMin",      tropmin_net),
    ("MultiAnchor",  anchor_net),
    ("MLP",          lambda: DeepMLP(num_layers=2)),
]
state = load()
for seed in [0, 1, 2]:
    for name, fac in configs:
        if done(state, seed, name): continue
        tr, va = make_loaders(n_train=3000, n_val=500, batch_size=64, seed=seed)
        torch.manual_seed(seed); m = fac()
        h = train_model(m, tr, va, epochs=10, lr=2e-3, name=f"{name}-s{seed}",
                        save_to=None, log_every=0, grad_clip=1.0)
        acc = float(max(h["val_acc"]))
        state["runs"].append({"seed": seed, "model": name, "best_val_acc": acc, "n_params": count_p(m)})
        save(state)
        print(f"  seed={seed} {name:13s} acc={acc:.4f} params={count_p(m)}")

print(f"\n=== Round 3 results ===")
print(f"{'model':>14s}  {'mean':>8s}  {'std':>8s}  {'gap':>9s}  {'wins':>5s}  params")
mlp = np.array([r["best_val_acc"] for r in state["runs"] if r["model"]=="MLP"])
for name, _ in configs:
    vals = np.array([r["best_val_acc"] for r in state["runs"] if r["model"]==name])
    n = min(len(vals), len(mlp))
    if n == 0: continue
    mu = vals[:n].mean(); sd = vals[:n].std(ddof=1) if n>1 else 0.0
    gap = (mu - mlp[:n].mean())*100 if name != "MLP" else 0.0
    wins = int((vals[:n] > mlp[:n]).sum()) if name != "MLP" else "-"
    npar = next(r["n_params"] for r in state["runs"] if r["model"]==name)
    print(f"{name:>14s}  {mu:>8.4f}  {sd:>8.4f}  {gap:>+8.2f}pp  {str(wins):>5s}  {npar}")
