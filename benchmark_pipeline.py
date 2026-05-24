"""benchmark_pipeline.py — rigorous benchmark of contrast-based FFN blocks.

Compares 5 FFN block types across 4 datasets, 2 depths, 5 seeds:

    Blocks:
        ReLU      —  y = x + W2 · ReLU(W1 x)                       (baseline)
        Bilinear  —  y = x + W_out · (W_a x ⊙ W_b x)               (Shazeer 2020)
        GLU       —  y = x + W_out · (W_a x ⊙ σ(W_b x))            (Dauphin 2017)
        Huber     —  y = x + W_out · huber(W_a x − W_b x)          (candidate)
        AbsDiff   —  y = x + W_out · |W_a x − W_b x|               (candidate)

    Datasets:
        Dynamical-4   — synthetic 4-class trajectories (64-dim signal)
        MNIST         — 10-class digits (784 flat)
        Fashion-MNIST — 10-class clothing (784 flat)
        CIFAR-10      — 10-class images (3072 flat — small FFN, no CNN encoder
                        intentionally, so we're testing the FFN block, not conv)

    Depths: L=2, L=4
    Seeds:  0..4

Everything saved incrementally to results/benchmark.json so the pipeline
resumes cleanly from interruption.

Hardware: auto-detects CUDA. Estimated runtime on a single GPU: ~1-3 hours.

Analyses:
  1. Per-(dataset, depth) mean±std with paired-seed win counts.
  2. Wilcoxon signed-rank test (Huber vs each baseline).
  3. Sample efficiency on Fashion-MNIST at n_train in {500, 2k, 10k, 50k}.
  4. Out-of-distribution check: train MNIST, test rotated MNIST.
  5. Wall-clock cost per block.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

RESULTS_DIR = Path(__file__).parent / "results" / "benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT = RESULTS_DIR / "benchmark.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[benchmark] device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[benchmark] GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ---------------------------------------------------------------------------
# Blocks (residual form: y = x + block_op(x))
# ---------------------------------------------------------------------------

class ReLUBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        with torch.no_grad(): self.fc2.weight.mul_(0.5)
    def forward(self, x): return x + self.fc2(F.relu(self.fc1(x)))


class BilinearBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.a = nn.Linear(dim, hidden_dim)
        self.b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        with torch.no_grad(): self.out.weight.mul_(0.5)
    def forward(self, x): return x + self.out(self.a(x) * self.b(x))


class GLUBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.a = nn.Linear(dim, hidden_dim)
        self.b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        with torch.no_grad(): self.out.weight.mul_(0.5)
    def forward(self, x): return x + self.out(self.a(x) * torch.sigmoid(self.b(x)))


class HuberBlock(nn.Module):
    def __init__(self, dim, hidden_dim, delta=1.0):
        super().__init__()
        self.a = nn.Linear(dim, hidden_dim)
        self.b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.delta = delta
        with torch.no_grad(): self.out.weight.mul_(0.5)
    def forward(self, x):
        d = self.a(x) - self.b(x)
        ad = d.abs()
        h = torch.where(ad < self.delta, 0.5 * d * d / self.delta, ad - 0.5 * self.delta)
        return x + self.out(h)


class AbsDiffBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.a = nn.Linear(dim, hidden_dim)
        self.b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        with torch.no_grad(): self.out.weight.mul_(0.5)
    def forward(self, x): return x + self.out(torch.abs(self.a(x) - self.b(x)))


BLOCKS = {
    "ReLU":     ReLUBlock,
    "Bilinear": BilinearBlock,
    "GLU":      GLUBlock,
    "Huber":    HuberBlock,
    "AbsDiff":  AbsDiffBlock,
}


# ---------------------------------------------------------------------------
# Matched-parameter sizing
# ---------------------------------------------------------------------------

def params_for_net(block_cls, input_dim, num_layers, dim, hidden_dim, num_classes):
    """Compute exact parameter count for a net with given block."""
    # encoder: input_dim*dim + dim
    p = input_dim * dim + dim
    # blocks: per block, depends on block type
    # ReLU: 2 linears + biases = dim*hidden + hidden + hidden*dim + dim
    # Bilinear/GLU/Huber/AbsDiff: 3 linears + biases (2 in, 1 out)
    #   = 2*(dim*hidden + hidden) + (hidden*dim + dim)
    if block_cls is ReLUBlock:
        per_block = 2 * dim * hidden_dim + hidden_dim + dim
    else:
        per_block = 3 * dim * hidden_dim + 2 * hidden_dim + dim
    p += num_layers * per_block
    # head: dim * num_classes + num_classes
    p += dim * num_classes + num_classes
    return p


def matched_hidden(block_cls, target_params, input_dim, num_layers, dim, num_classes):
    """Find hidden_dim such that net's param count <= target_params (closest)."""
    best, best_diff = None, math.inf
    for h in range(4, 1024):
        n = params_for_net(block_cls, input_dim, num_layers, dim, h, num_classes)
        if n <= target_params:
            best, best_diff = h, target_params - n
        else:
            if best is not None and n - target_params < best_diff:
                # n is closer overshooting
                return h
            return best if best is not None else h
    return best


class GenericNet(nn.Module):
    def __init__(self, block_cls, input_dim, num_layers, dim, hidden_dim, num_classes):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([block_cls(dim, hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, num_classes)
    def forward(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        for b in self.blocks: h = b(h)
        return self.head(h)


def make_model(block_name, input_dim, num_layers, dim, num_classes, target_params):
    block_cls = BLOCKS[block_name]
    h = matched_hidden(block_cls, target_params, input_dim, num_layers, dim, num_classes)
    return GenericNet(block_cls, input_dim, num_layers, dim, h, num_classes), h


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    name: str
    input_dim: int
    num_classes: int
    train_loader_fn: Callable        # (seed, n_train=None) -> (train, val)


def _torchvision_loaders(ds_cls, mean, std, in_dim, batch_size=256, seed=0, train_subset=None):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    root = "./data"
    Path(root).mkdir(exist_ok=True)
    tr = ds_cls(root, train=True, download=True, transform=tfm)
    va = ds_cls(root, train=False, download=True, transform=tfm)
    if train_subset is not None and train_subset < len(tr):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(tr))[:train_subset]
        tr = Subset(tr, idx.tolist())
    g = torch.Generator(); g.manual_seed(seed)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g,
                       drop_last=True, num_workers=2, pin_memory=DEVICE.type=="cuda"),
            DataLoader(va, batch_size=batch_size, shuffle=False, drop_last=False,
                       num_workers=2, pin_memory=DEVICE.type=="cuda"))


def mnist_loaders(seed=0, train_subset=None):
    return _torchvision_loaders(datasets.MNIST, (0.1307,), (0.3081,), 784,
                                seed=seed, train_subset=train_subset)

def fashion_loaders(seed=0, train_subset=None):
    return _torchvision_loaders(datasets.FashionMNIST, (0.286,), (0.353,), 784,
                                seed=seed, train_subset=train_subset)

def cifar_loaders(seed=0, train_subset=None):
    return _torchvision_loaders(datasets.CIFAR10,
                                (0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616), 3072,
                                seed=seed, train_subset=train_subset)


def dynamical_loaders(seed=0, train_subset=None):
    from dynamical_data import make_loaders
    n = train_subset if train_subset is not None else 5000
    return make_loaders(n_train=n, n_val=1000, batch_size=256, seed=seed)


DATASETS = {
    "Dynamical-4":   DatasetSpec("Dynamical-4",   64,   4, dynamical_loaders),
    "MNIST":         DatasetSpec("MNIST",         784, 10, mnist_loaders),
    "Fashion-MNIST": DatasetSpec("Fashion-MNIST", 784, 10, fashion_loaders),
    "CIFAR-10":      DatasetSpec("CIFAR-10",      3072,10, cifar_loaders),
}


# ---------------------------------------------------------------------------
# Train & eval
# ---------------------------------------------------------------------------

def train_eval(model, train_loader, val_loader, epochs=20, lr=2e-3, wd=1e-4):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    train_t = 0.0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        train_t += time.time() - t0
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                p = model(x).argmax(-1)
                correct += (p == y).sum().item(); total += y.numel()
        acc = correct / total
        best_val_acc = max(best_val_acc, acc)
    return best_val_acc, train_t


# ---------------------------------------------------------------------------
# Runner / state
# ---------------------------------------------------------------------------

def load_state():
    if OUT.exists(): return json.loads(OUT.read_text())
    return {"runs": []}

def save_state(s): OUT.write_text(json.dumps(s, indent=2))

def already_done(state, **kw):
    return any(all(r.get(k) == v for k, v in kw.items()) for r in state["runs"])


# ---------------------------------------------------------------------------
# Experiment 1: full grid (datasets × blocks × depths × seeds)
# ---------------------------------------------------------------------------

def experiment_grid(state, datasets_to_run, blocks_to_run, depths, seeds,
                    target_params=None, epochs=20):
    """Full crossed design."""
    for ds_name in datasets_to_run:
        ds = DATASETS[ds_name]
        # Pick a target param count if not supplied: pick "what ReLU L=4 with hidden=128 needs"
        if target_params is None:
            target = params_for_net(ReLUBlock, ds.input_dim, 4, 64, 128, ds.num_classes)
        else:
            target = target_params
        for L in depths:
            for seed in seeds:
                # Same train/val loader per seed across blocks: ensure FAIR data shuffles
                tr, va = ds.train_loader_fn(seed=seed)
                for block_name in blocks_to_run:
                    if already_done(state, experiment="grid", dataset=ds_name,
                                    block=block_name, depth=L, seed=seed):
                        continue
                    torch.manual_seed(seed)
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                    model, h_dim = make_model(block_name, ds.input_dim, L, 64,
                                              ds.num_classes, target)
                    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    t0 = time.time()
                    acc, train_t = train_eval(model, tr, va, epochs=epochs)
                    state["runs"].append({
                        "experiment": "grid", "dataset": ds_name, "block": block_name,
                        "depth": L, "seed": seed, "hidden_dim": h_dim,
                        "n_params": n_params, "best_val_acc": float(acc),
                        "wall_time_s": time.time() - t0, "train_time_s": train_t,
                    })
                    save_state(state)
                    print(f"  [{ds_name:13s} L={L} {block_name:8s} s={seed}] "
                          f"acc={acc:.4f}  params={n_params}  "
                          f"hidden={h_dim}  time={time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Experiment 2: sample efficiency on Fashion-MNIST
# ---------------------------------------------------------------------------

def experiment_sample_efficiency(state, blocks_to_run, sizes, seeds, depth=2, epochs=20):
    ds = DATASETS["Fashion-MNIST"]
    target = params_for_net(ReLUBlock, ds.input_dim, depth, 64, 128, ds.num_classes)
    for n in sizes:
        for seed in seeds:
            tr, va = ds.train_loader_fn(seed=seed, train_subset=n)
            for block_name in blocks_to_run:
                if already_done(state, experiment="sample_eff", n=n, block=block_name,
                                seed=seed):
                    continue
                torch.manual_seed(seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                model, h_dim = make_model(block_name, ds.input_dim, depth, 64,
                                          ds.num_classes, target)
                t0 = time.time()
                acc, train_t = train_eval(model, tr, va, epochs=epochs)
                state["runs"].append({
                    "experiment": "sample_eff", "n": n, "block": block_name,
                    "seed": seed, "best_val_acc": float(acc),
                    "wall_time_s": time.time() - t0,
                })
                save_state(state)
                print(f"  [SE n={n:>5d} {block_name:8s} s={seed}] acc={acc:.4f} "
                      f"time={time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Experiment 3: OOD robustness — train MNIST, test rotated/noisy MNIST
# ---------------------------------------------------------------------------

class RotatedMNIST(Dataset):
    def __init__(self, base_dataset, degrees=15.0):
        self.base = base_dataset
        self.rot = transforms.RandomRotation((degrees, degrees))
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return self.rot(x), y


def experiment_ood(state, blocks_to_run, seeds, depth=4, epochs=20):
    """Train on standard MNIST, evaluate on rotated MNIST (15°, 30°, 45°)."""
    ds = DATASETS["MNIST"]
    target = params_for_net(ReLUBlock, ds.input_dim, depth, 64, 128, ds.num_classes)
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    val_base = datasets.MNIST("./data", train=False, download=True, transform=tfm)

    for seed in seeds:
        tr, va = ds.train_loader_fn(seed=seed)
        for block_name in blocks_to_run:
            if already_done(state, experiment="ood", block=block_name, seed=seed):
                continue
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            model, h_dim = make_model(block_name, ds.input_dim, depth, 64,
                                      ds.num_classes, target)
            acc_clean, _ = train_eval(model, tr, va, epochs=epochs)
            # rotated evals
            ood = {}
            for deg in [15.0, 30.0, 45.0]:
                rot_ds = RotatedMNIST(val_base, degrees=deg)
                rl = DataLoader(rot_ds, batch_size=256, num_workers=2,
                                pin_memory=DEVICE.type=="cuda")
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for x, y in rl:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        correct += (model(x).argmax(-1) == y).sum().item()
                        total += y.numel()
                ood[deg] = correct / total
            state["runs"].append({
                "experiment": "ood", "block": block_name, "seed": seed,
                "clean_acc": float(acc_clean),
                "rot_15": float(ood[15.0]), "rot_30": float(ood[30.0]), "rot_45": float(ood[45.0]),
            })
            save_state(state)
            print(f"  [OOD {block_name:8s} s={seed}] clean={acc_clean:.4f}  "
                  f"rot15={ood[15.0]:.4f}  rot30={ood[30.0]:.4f}  rot45={ood[45.0]:.4f}")


# ---------------------------------------------------------------------------
# Analysis & reporting
# ---------------------------------------------------------------------------

def report(state):
    print("\n" + "=" * 78)
    print("BENCHMARK REPORT")
    print("=" * 78)

    # ---- Experiment 1: grid ----
    grid = [r for r in state["runs"] if r["experiment"] == "grid"]
    if grid:
        # Per (dataset, depth, block) mean ± std
        print("\n## Experiment 1 — main grid (mean val acc ± std over seeds)\n")
        ds_set = sorted({r["dataset"] for r in grid})
        depths = sorted({r["depth"] for r in grid})
        blocks = ["ReLU", "Bilinear", "GLU", "Huber", "AbsDiff"]
        for ds_name in ds_set:
            for L in depths:
                rows = [r for r in grid if r["dataset"] == ds_name and r["depth"] == L]
                if not rows: continue
                print(f"\n  {ds_name} (L={L}):")
                relu_vals = np.array([r["best_val_acc"] for r in rows if r["block"] == "ReLU"])
                bilin_vals = np.array([r["best_val_acc"] for r in rows if r["block"] == "Bilinear"])
                ref = relu_vals
                for blk in blocks:
                    vals = np.array([r["best_val_acc"] for r in rows if r["block"] == blk])
                    if len(vals) == 0: continue
                    n = min(len(vals), len(ref))
                    v, r0 = vals[:n], ref[:n]
                    wins = int((v > r0).sum()) if blk != "ReLU" else "—"
                    gap = (v.mean() - r0.mean()) * 100 if blk != "ReLU" else 0.0
                    pval = "—"
                    if blk != "ReLU" and wilcoxon and n >= 5:
                        try:
                            pv = wilcoxon(v, r0, alternative="greater").pvalue
                            pval = f"{pv:.3f}"
                        except ValueError:
                            pass
                    print(f"    {blk:>10s}:  {v.mean():.4f} ± {v.std(ddof=1):.4f}  "
                          f"gap_vs_ReLU={gap:+5.2f}pp  wins={wins}/{n}  p={pval}")
                # Also vs Bilinear
                if len(bilin_vals) > 0:
                    for blk in ["Huber", "AbsDiff"]:
                        vals = np.array([r["best_val_acc"] for r in rows if r["block"] == blk])
                        n = min(len(vals), len(bilin_vals))
                        if n == 0: continue
                        v, ref2 = vals[:n], bilin_vals[:n]
                        wins = int((v > ref2).sum())
                        gap = (v.mean() - ref2.mean()) * 100
                        pval = "—"
                        if wilcoxon and n >= 5:
                            try:
                                pv = wilcoxon(v, ref2, alternative="greater").pvalue
                                pval = f"{pv:.3f}"
                            except ValueError:
                                pass
                        print(f"        ({blk} vs Bilinear: gap={gap:+5.2f}pp  wins={wins}/{n}  p={pval})")

        # Cross-dataset summary
        print(f"\n## Cross-dataset summary (mean across seeds per dataset, depth=4)\n")
        print(f"  {'block':>10s}  {'Dyn-4':>8s}  {'MNIST':>8s}  {'Fashion':>8s}  {'CIFAR':>8s}")
        for blk in blocks:
            row = [blk]
            for ds_name in ["Dynamical-4", "MNIST", "Fashion-MNIST", "CIFAR-10"]:
                vals = [r["best_val_acc"] for r in grid
                        if r["dataset"] == ds_name and r["depth"] == 4 and r["block"] == blk]
                row.append(f"{np.mean(vals):.4f}" if vals else "  —  ")
            print(f"  {row[0]:>10s}  {row[1]:>8s}  {row[2]:>8s}  {row[3]:>8s}  {row[4]:>8s}")

    # ---- Experiment 2: sample efficiency ----
    se = [r for r in state["runs"] if r["experiment"] == "sample_eff"]
    if se:
        print(f"\n## Experiment 2 — Fashion-MNIST sample efficiency (mean val acc)\n")
        sizes = sorted({r["n"] for r in se})
        blocks = sorted({r["block"] for r in se})
        print(f"  {'n':>6s}  " + "  ".join(f"{b:>8s}" for b in blocks))
        for n in sizes:
            row = [f"{n:>6d}"]
            for b in blocks:
                v = [r["best_val_acc"] for r in se if r["n"] == n and r["block"] == b]
                row.append(f"{np.mean(v):.4f}" if v else "  —  ")
            print("  " + "  ".join(row))

    # ---- Experiment 3: OOD ----
    ood = [r for r in state["runs"] if r["experiment"] == "ood"]
    if ood:
        print(f"\n## Experiment 3 — OOD robustness (rotated MNIST, mean val acc)\n")
        blocks = sorted({r["block"] for r in ood})
        print(f"  {'block':>10s}  {'clean':>8s}  {'rot15':>8s}  {'rot30':>8s}  {'rot45':>8s}")
        for blk in blocks:
            rows = [r for r in ood if r["block"] == blk]
            if not rows: continue
            clean = np.mean([r["clean_acc"] for r in rows])
            r15 = np.mean([r["rot_15"] for r in rows])
            r30 = np.mean([r["rot_30"] for r in rows])
            r45 = np.mean([r["rot_45"] for r in rows])
            print(f"  {blk:>10s}  {clean:.4f}  {r15:.4f}  {r30:.4f}  {r45:.4f}")

    # ---- Cost ----
    print(f"\n## Wall-clock cost (average per-run, seconds)\n")
    blocks = sorted({r.get("block") for r in state["runs"] if r["experiment"] == "grid"})
    for blk in blocks:
        rows = [r for r in state["runs"] if r["experiment"] == "grid" and r.get("block") == blk]
        if rows:
            print(f"  {blk:>10s}: {np.mean([r['wall_time_s'] for r in rows]):.1f}s  "
                  f"(train-only: {np.mean([r['train_time_s'] for r in rows]):.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="just print report from saved state")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--skip-se", action="store_true")
    ap.add_argument("--skip-ood", action="store_true")
    ap.add_argument("--datasets", nargs="+",
                    default=["Dynamical-4", "MNIST", "Fashion-MNIST", "CIFAR-10"])
    ap.add_argument("--blocks", nargs="+",
                    default=["ReLU", "Bilinear", "GLU", "Huber", "AbsDiff"])
    ap.add_argument("--depths", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    state = load_state()
    if args.report:
        report(state); return

    print(f"[benchmark] previous runs: {len(state['runs'])}")
    print(f"[benchmark] config: datasets={args.datasets}  blocks={args.blocks}  "
          f"depths={args.depths}  seeds={args.seeds}  epochs={args.epochs}")

    if not args.skip_grid:
        print("\n=== Experiment 1: main grid ===")
        experiment_grid(state, args.datasets, args.blocks, args.depths,
                        args.seeds, epochs=args.epochs)

    if not args.skip_se:
        print("\n=== Experiment 2: sample efficiency on Fashion-MNIST ===")
        experiment_sample_efficiency(state, args.blocks,
                                     sizes=[500, 2000, 10000, 50000],
                                     seeds=args.seeds, depth=2, epochs=args.epochs)

    if not args.skip_ood:
        print("\n=== Experiment 3: OOD robustness (rotated MNIST) ===")
        experiment_ood(state, args.blocks, args.seeds, depth=4, epochs=args.epochs)

    report(state)


if __name__ == "__main__":
    main()
