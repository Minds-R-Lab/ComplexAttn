"""ResNet-18-Huber vs ResNet-18 on CIFAR-10.

This is the SCALE test: does the small-scale Huber win transfer to a real
deep CNN? Multi-seed, matched params (or matched FLOPs — pick one).

Usage:
    python resnet_huber.py                   # full multi-seed run
    python resnet_huber.py --quick           # 1 seed, 20 epochs, sanity check
    python resnet_huber.py --report          # just summarize saved results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, BasicBlock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path(__file__).parent / "results" / "resnet_huber"
RESULTS.mkdir(parents=True, exist_ok=True)
OUT = RESULTS / "resnet_huber.json"


# ---------------------------------------------------------------------------
# BasicHuberBlock — drop-in replacement for BasicBlock
# ---------------------------------------------------------------------------

class BasicHuberBlock(nn.Module):
    """Two-branch contrast block matching the spatial shape of BasicBlock."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 delta=1.0):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64 or dilation != 1:
            raise ValueError("BasicHuberBlock only supports groups=1, base_width=64, dilation=1")
        self.conv_a = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn_a   = norm_layer(planes)
        self.conv_b = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn_b   = norm_layer(planes)
        self.conv_out = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn_out   = norm_layer(planes)
        self.downsample = downsample
        self.delta = delta

    def _huber(self, d):
        ad = d.abs()
        return torch.where(ad < self.delta,
                           0.5 * d * d / self.delta,
                           ad - 0.5 * self.delta)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        a = self.bn_a(self.conv_a(x))
        b = self.bn_b(self.conv_b(x))
        out = self._huber(a - b)
        out = self.bn_out(self.conv_out(out))
        return F.relu(out + identity)


def resnet18(num_classes=10, planes_scale=1.0):
    """Standard ResNet-18 from torchvision, with optional channel scaling for
    matching param count of the Huber variant."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if planes_scale != 1.0:
        raise NotImplementedError("Use the standard layer widths; match params via Huber-side scaling")
    return model


def resnet18_huber(num_classes=10, planes_scale=1.0):
    """ResNet-18 with BasicHuberBlock replacing BasicBlock."""
    model = ResNet(BasicHuberBlock, [2, 2, 2, 2], num_classes=num_classes)
    if planes_scale < 1.0:
        # Optionally shrink all conv layers proportionally to match params with ResNet18
        # (you'd need to subclass ResNet to do this cleanly; skip for now)
        raise NotImplementedError
    return model


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# CIFAR-10 data
# ---------------------------------------------------------------------------

def cifar10_loaders(seed=0, batch_size=128):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    root = "./data"
    Path(root).mkdir(exist_ok=True)
    tr = datasets.CIFAR10(root, train=True, download=True, transform=train_tfm)
    va = datasets.CIFAR10(root, train=False, download=True, transform=test_tfm)
    g = torch.Generator(); g.manual_seed(seed)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g,
                       num_workers=4, pin_memory=DEVICE.type=="cuda"),
            DataLoader(va, batch_size=512, shuffle=False,
                       num_workers=4, pin_memory=DEVICE.type=="cuda"))


# ---------------------------------------------------------------------------
# Train & eval
# ---------------------------------------------------------------------------

def train_eval(model, train_loader, val_loader, epochs=80, lr=0.1, wd=5e-4):
    model.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y); loss.backward(); opt.step()
        sched.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                correct += (model(x).argmax(-1) == y).sum().item(); total += y.numel()
        acc = correct / total
        best = max(best, acc)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    epoch {ep+1}/{epochs}  val_acc={acc:.4f}  best={best:.4f}")
    return best


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_state():
    return json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}

def save_state(s): OUT.write_text(json.dumps(s, indent=2))

def done(state, seed, model):
    return any(r["seed"]==seed and r["model"]==model for r in state["runs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, 20 epochs")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    state = load_state()
    if args.report:
        from statistics import mean, stdev
        import numpy as np
        relu = [r["best_val_acc"] for r in state["runs"] if r["model"]=="ResNet18"]
        huber = [r["best_val_acc"] for r in state["runs"] if r["model"]=="ResNet18-Huber"]
        n = min(len(relu), len(huber))
        if n == 0:
            print("no completed runs"); return
        r, h = np.array(relu[:n]), np.array(huber[:n])
        print(f"ResNet18:        {r.mean():.4f} ± {r.std(ddof=1):.4f}  per-seed: {r.round(4).tolist()}")
        print(f"ResNet18-Huber:  {h.mean():.4f} ± {h.std(ddof=1):.4f}  per-seed: {h.round(4).tolist()}")
        print(f"Gap:             {(h.mean()-r.mean())*100:+.2f}pp   Huber wins paired: {int((h>r).sum())}/{n}")
        return

    seeds = [0] if args.quick else args.seeds
    epochs = 20 if args.quick else args.epochs

    print(f"device: {DEVICE}")
    print(f"params: ResNet18 = {count_params(resnet18())}   "
          f"ResNet18-Huber = {count_params(resnet18_huber())}   "
          f"(ratio: {count_params(resnet18_huber()) / count_params(resnet18()):.2f}x)")

    for seed in seeds:
        for name, factory in [("ResNet18", resnet18), ("ResNet18-Huber", resnet18_huber)]:
            if done(state, seed, name): continue
            print(f"\n--- {name} seed={seed} ---")
            torch.manual_seed(seed)
            if DEVICE.type == "cuda": torch.cuda.manual_seed_all(seed)
            tr, va = cifar10_loaders(seed=seed)
            model = factory()
            t0 = time.time()
            acc = train_eval(model, tr, va, epochs=epochs)
            state["runs"].append({
                "seed": seed, "model": name, "best_val_acc": float(acc),
                "epochs": epochs, "wall_time_s": time.time() - t0,
                "n_params": count_params(model),
            })
            save_state(state)
            print(f"  -> best val acc {acc:.4f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
