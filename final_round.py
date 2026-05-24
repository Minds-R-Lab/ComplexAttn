"""Final round: stress-test AbsDiff and two variants at L=2 and L=4, 5 seeds.

Variants:
  - AbsDiff: y = x + W_out · |W_a x - W_b x|        (piecewise linear contrast)
  - SoftAbs: y = x + W_out · sqrt((W_a x - W_b x)^2 + eps)  (smooth L1)
  - Huber:   y = x + W_out · huber(W_a x - W_b x)   (smooth near 0, linear far)
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from dynamical_data import make_loaders
from novel_blocks import AbsDiffBlock, GenericBlockNet
from stacked import DeepMLP
from train import train_model


class SoftAbsBlock(nn.Module):
    def __init__(self, dim, hidden_dim, eps=1e-3):
        super().__init__()
        self.branch_a = nn.Linear(dim, hidden_dim)
        self.branch_b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.eps = eps
        with torch.no_grad(): self.out.weight.mul_(0.5)
    def forward(self, x):
        d = self.branch_a(x) - self.branch_b(x)
        return x + self.out(torch.sqrt(d*d + self.eps))


class HuberBlock(nn.Module):
    def __init__(self, dim, hidden_dim, delta=1.0):
        super().__init__()
        self.branch_a = nn.Linear(dim, hidden_dim)
        self.branch_b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.delta = delta
        with torch.no_grad(): self.out.weight.mul_(0.5)
    def forward(self, x):
        d = self.branch_a(x) - self.branch_b(x)
        abs_d = d.abs()
        huber = torch.where(abs_d < self.delta,
                            0.5 * d * d / self.delta,
                            abs_d - 0.5 * self.delta)
        return x + self.out(huber)


def absdiff_net(L):  return GenericBlockNet(lambda d: AbsDiffBlock(d, 16), num_layers=L)
def softabs_net(L):  return GenericBlockNet(lambda d: SoftAbsBlock(d, 16),  num_layers=L)
def huber_net(L):    return GenericBlockNet(lambda d: HuberBlock(d, 16),    num_layers=L)


OUT = Path("results/final_round.json")
def load(): return json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}
def save(s): OUT.write_text(json.dumps(s, indent=2))
def done(state, L, seed, name):
    return any(r["L"]==L and r["seed"]==seed and r["model"]==name for r in state["runs"])

torch.set_num_threads(1)
def count_p(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
configs = [
    ("AbsDiff", absdiff_net),
    ("SoftAbs", softabs_net),
    ("Huber",   huber_net),
    ("MLP",     lambda L: DeepMLP(num_layers=L)),
]
state = load()
for L in [2, 4]:
    for seed in [0, 1, 2, 3, 4]:
        for name, fac in configs:
            if done(state, L, seed, name): continue
            tr, va = make_loaders(n_train=3000, n_val=500, batch_size=64, seed=seed)
            torch.manual_seed(seed); m = fac(L)
            h = train_model(m, tr, va, epochs=10, lr=2e-3, name=f"{name}-L{L}-s{seed}",
                            save_to=None, log_every=0, grad_clip=1.0)
            acc = float(max(h["val_acc"]))
            state["runs"].append({"L": L, "seed": seed, "model": name,
                                   "best_val_acc": acc, "n_params": count_p(m)})
            save(state)
            print(f"  L={L} seed={seed} {name:8s} acc={acc:.4f} params={count_p(m)}")

print(f"\n=== Final round: 5 seeds, L=2 and L=4 ===")
print(f"{'L':>3s}  {'model':>8s}  {'mean±std':>16s}  {'gap':>10s}  {'wins':>6s}  params")
for L in [2, 4]:
    mlp = np.array([r["best_val_acc"] for r in state["runs"] if r["L"]==L and r["model"]=="MLP"])
    for name, _ in configs:
        vals = np.array([r["best_val_acc"] for r in state["runs"] if r["L"]==L and r["model"]==name])
        n = min(len(vals), len(mlp))
        if n == 0: continue
        mu = vals[:n].mean(); sd = vals[:n].std(ddof=1) if n>1 else 0.0
        gap = (mu - mlp[:n].mean())*100 if name != "MLP" else 0.0
        wins = int((vals[:n] > mlp[:n]).sum()) if name != "MLP" else "-"
        np_ = next(r["n_params"] for r in state["runs"] if r["L"]==L and r["model"]==name)
        print(f"{L:>3d}  {name:>8s}  {mu:.4f}±{sd:.4f}  {gap:>+8.2f}pp  {str(wins):>5s}  {np_}")
