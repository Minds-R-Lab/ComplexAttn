"""Test novel blocks against MLP on dynamical task. 3 seeds, 10 epochs, L=2."""
import json
from pathlib import Path
import numpy as np
import torch

from dynamical_data import make_loaders
from novel_blocks import absdiff_net, antisym_net, cubic_net, sqdiff_net, count_trainable
from stacked import DeepMLP
from train import train_model

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "novel.json"

def load(): return json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}
def save(s): OUT.write_text(json.dumps(s, indent=2))
def done(state, seed, name):
    return any(r["seed"]==seed and r["model"]==name for r in state["runs"])

def run_one(seed, name, factory, lr=2e-3, epochs=10):
    tr, va = make_loaders(n_train=3000, n_val=500, batch_size=64, seed=seed)
    torch.manual_seed(seed); m = factory()
    h = train_model(m, tr, va, epochs=epochs, lr=lr, name=f"{name}-s{seed}",
                    save_to=None, log_every=0, grad_clip=1.0)
    return float(max(h["val_acc"])), count_trainable(m)

torch.set_num_threads(1)
seeds = [0, 1, 2]
configs = [
    ("AbsDiff", lambda: absdiff_net(num_layers=2)),
    ("Antisym", lambda: antisym_net(num_layers=2)),
    ("Cubic",   lambda: cubic_net(num_layers=2)),
    ("SqDiff",  lambda: sqdiff_net(num_layers=2)),
    ("MLP",     lambda: DeepMLP(num_layers=2)),
]
state = load()
for seed in seeds:
    for name, factory in configs:
        if done(state, seed, name): continue
        acc, npar = run_one(seed, name, factory)
        state["runs"].append({"seed": seed, "model": name, "best_val_acc": acc, "n_params": npar})
        save(state)
        print(f"  seed={seed} {name:8s} acc={acc:.4f} params={npar}")

print(f"\n=== Novel-block comparison, dynamical L=2, 3 seeds × 10 epochs ===")
print(f"{'model':>8s}  {'mean':>8s}  {'std':>8s}  {'gap vs MLP':>11s}  {'wins':>5s}  params")
mlp_vals = np.array([r["best_val_acc"] for r in state["runs"] if r["model"]=="MLP"])
for name, _ in configs:
    vals = np.array([r["best_val_acc"] for r in state["runs"] if r["model"]==name])
    n = min(len(vals), len(mlp_vals))
    if n == 0: continue
    mu = vals[:n].mean(); sd = vals[:n].std(ddof=1) if n>1 else 0.0
    gap = (mu - mlp_vals[:n].mean()) * 100 if name != "MLP" else 0.0
    wins = int((vals[:n] > mlp_vals[:n]).sum()) if name != "MLP" else "-"
    npar = next(r["n_params"] for r in state["runs"] if r["model"]==name)
    print(f"{name:>8s}  {mu:>8.4f}  {sd:>8.4f}  {gap:>+10.2f}pp  {str(wins):>5s}  {npar}")
