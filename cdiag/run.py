"""
Training and evaluation for the real vs complex diagonal SSM copy-task
comparison.

The experiment compares three settings:

  Setting A:  RealDiagSSM(n_state=N)    vs  ComplexDiagSSM(n_state=N)
    Matched STATE dimension. Complex has 2x the parameters.

  Setting B:  RealDiagSSM(n_state=2*N)  vs  ComplexDiagSSM(n_state=N)
    Matched (roughly) PARAMETER COUNT. Real has the same number of
    real-valued parameters as complex.

  Setting C:  Various N for real, fixed N for complex. Sweep.
    Map out: at what N does real catch up to complex?

We track several things during training:
  - cross-entropy loss
  - token accuracy on the output positions
  - L2 norm of model parameters (Ran-Milo predicts blow-up for real)
  - magnitudes of learned lambdas (do they push toward |lam|=1?)
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import RealDiagSSM, ComplexDiagSSM, RealRoPESSM
from data import make_batch


def evaluate(model, K, L, A, n_batches=4, batch_size=64, device="cpu", seed=12345):
    """Compute token-level accuracy on output positions."""
    model.eval()
    correct, total = 0, 0
    losses = []
    with torch.no_grad():
        for i in range(n_batches):
            x, y, mask = make_batch(batch_size, K, L, A, device=device,
                                     seed=seed + i)
            logits = model(x)                  # [B, T, A+1]
            loss = F.cross_entropy(
                logits[mask], y[mask], reduction="mean")
            losses.append(loss.item())
            pred = logits.argmax(-1)
            correct += int(((pred == y) & mask).sum().item())
            total   += int(mask.sum().item())
    model.train()
    return correct / max(total, 1), float(np.mean(losses))


def train(model, K, L, A, lr, steps, batch_size, device, log_every=200,
          val_every=200, tag=""):
    """Train one model on the copy task and return the training history."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = {
        "step": [], "train_loss": [], "val_acc": [], "val_loss": [],
        "param_norm": [], "lam_max_mag": [], "lam_mean_mag": [],
    }
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        x, y, mask = make_batch(batch_size, K, L, A, device=device)
        logits = model(x)                       # [B, T, A+1]
        loss = F.cross_entropy(logits[mask], y[mask])
        opt.zero_grad()
        loss.backward()
        # Gradient clipping is essential for the real model -- it
        # blows up otherwise. Use a generous clip so we don't suppress
        # real signal.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            lam = model.lam.detach()
            mag = lam.abs() if lam.is_complex() else lam.abs()
            with torch.no_grad():
                p_norm = model.param_l2_norm().item()
            hist["step"].append(step)
            hist["train_loss"].append(loss.item())
            hist["param_norm"].append(p_norm)
            hist["lam_max_mag"].append(float(mag.max().item()))
            hist["lam_mean_mag"].append(float(mag.mean().item()))
            if step % val_every == 0 or step == 1:
                acc, vloss = evaluate(model, K, L, A, device=device)
                hist["val_acc"].append(acc)
                hist["val_loss"].append(vloss)
                print(f"[{tag}] step {step:>5d}/{steps}  "
                      f"train_loss={loss.item():.4f}  "
                      f"val_acc={acc:.3f}  "
                      f"||theta||={p_norm:.2f}  "
                      f"max|lam|={mag.max().item():.4f}")
            else:
                # Pad val arrays so length stays aligned with step array
                hist["val_acc"].append(hist["val_acc"][-1] if hist["val_acc"] else 0.0)
                hist["val_loss"].append(hist["val_loss"][-1] if hist["val_loss"] else float("inf"))
    hist["wallclock_sec"] = time.time() - t0
    print(f"[{tag}] done in {hist['wallclock_sec']:.1f}s")
    return hist


# ============================================================
# Driver
# ============================================================

def make_model(kind: str, n_state: int, d_in: int, d_out: int):
    if kind == "real":
        return RealDiagSSM(n_state, d_in, d_out)
    elif kind == "complex":
        return ComplexDiagSSM(n_state, d_in, d_out)
    elif kind == "real_rope":
        return RealRoPESSM(n_state, d_in, d_out, learn_theta=True)
    elif kind == "real_rope_frozen":
        return RealRoPESSM(n_state, d_in, d_out, learn_theta=False)
    elif kind == "real_rope_neg":
        return RealRoPESSM(n_state, d_in, d_out, learn_theta=True,
                            allow_negative_decay=True)
    raise ValueError(kind)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_one(kind, n_state, K, L, A, lr, steps, batch_size, device, seed, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d_in = A + 2     # one-hot input
    d_out = A + 1    # output classes
    model = make_model(kind, n_state, d_in, d_out).to(device)
    print(f"\n[{tag}]  kind={kind}  n_state={n_state}  "
          f"params={n_params(model):,}  device={device}")
    hist = train(model, K, L, A, lr, steps, batch_size, device, tag=tag)
    final_acc, final_loss = evaluate(model, K, L, A, device=device,
                                       n_batches=8, batch_size=128)
    return {
        "kind": kind, "n_state": n_state, "seed": seed,
        "params": n_params(model),
        "K": K, "L": L, "A": A, "lr": lr, "steps": steps,
        "history": hist,
        "final_acc": final_acc, "final_loss": final_loss,
    }


def plot_comparison(rows, outpath, title_suffix=""):
    """Two-panel plot: training loss and validation accuracy vs step."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for row in rows:
        kind = row["kind"]
        steps = row["history"]["step"]
        tloss = row["history"]["train_loss"]
        vacc  = row["history"]["val_acc"]
        color = "C0" if kind == "complex" else "C3"
        label = f"{kind}  n_state={row['n_state']}  seed={row['seed']}  ({row['params']:,} params)"
        axes[0].plot(steps, tloss, color=color, alpha=0.7, label=label, lw=1.5)
        axes[1].plot(steps, vacc,  color=color, alpha=0.7, label=label, lw=1.5)
    axes[0].set_xlabel("Training step"); axes[0].set_ylabel("Train cross-entropy")
    axes[0].set_yscale("log"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7)
    axes[0].set_title("Training loss")
    axes[1].set_xlabel("Training step"); axes[1].set_ylabel("Token accuracy")
    axes[1].set_ylim(-0.02, 1.05); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7)
    axes[1].set_title("Validation token accuracy")
    fig.suptitle(f"Real vs Complex Diagonal SSM on the Copy task{title_suffix}")
    plt.tight_layout(); plt.savefig(outpath, dpi=130); plt.close()


def plot_param_dynamics(rows, outpath):
    """Track parameter L2 norm and lambda magnitude over training."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for row in rows:
        kind = row["kind"]
        steps = row["history"]["step"]
        pnorm = row["history"]["param_norm"]
        lmax  = row["history"]["lam_max_mag"]
        color = "C0" if kind == "complex" else "C3"
        label = f"{kind} n={row['n_state']} s{row['seed']}"
        axes[0].plot(steps, pnorm, color=color, alpha=0.7, lw=1.5, label=label)
        axes[1].plot(steps, lmax,  color=color, alpha=0.7, lw=1.5, label=label)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("L2 norm of all parameters")
    axes[0].set_yscale("log")
    axes[0].set_title("Parameter L2 norm during training")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7)
    axes[1].axhline(1.0, color="k", ls=":", lw=0.7, label="stability boundary")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel(r"$\max_i |\lambda_i|$")
    axes[1].set_title("Largest eigenvalue magnitude")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(outpath, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full",
                    choices=("smoke", "matched_n", "full", "L_sweep",
                              "rope_test"))
    ap.add_argument("--outdir", default="results_cdiag")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--L", type=int, default=50)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    A = 8   # vocabulary size for data tokens
    lr = 5e-3
    batch_size = 64

    if args.config == "smoke":
        configs = [("real", 16), ("complex", 16)]
        seeds = [0]
        steps = 400
    elif args.config == "matched_n":
        # Matched state dim; complex has 2x params
        configs = [("real", 16), ("complex", 16), ("real", 32), ("complex", 32)]
        seeds = list(range(args.seeds))
        steps = args.steps
    elif args.config == "full":
        # Real at varying n_state to see if it catches up; complex at n=16 only
        configs = [
            ("complex", 16),    # baseline
            ("real",    16),    # matched state dim
            ("real",    32),    # 2x state dim, matched real params
            ("real",    64),    # 4x state dim
            ("real",   128),    # 8x state dim
        ]
        seeds = list(range(args.seeds))
        steps = args.steps
    elif args.config == "L_sweep":
        # PROPERLY DESIGNED comparison:
        # Fix complex n_state to be enough that complex solves the task.
        # Then scale real to many sizes and see if/where it catches up.
        # The Ran-Milo prediction: real should plateau below 100%
        # regardless of n_state.
        configs  = [("complex", 16)]
        # Real sized aggressively: up to 32x the complex model.
        configs += [("real", n) for n in [16, 32, 64, 128, 256, 512]]
        seeds = list(range(args.seeds))
        steps = args.steps
    elif args.config == "rope_test":
        # Test (A) from the project notes: does Mamba-3-style RoPE on
        # real-decay SSM rescue accuracy and break kernel collapse?
        # Compare:
        #   real            -- the failing baseline
        #   complex         -- the succeeding baseline
        #   real_rope       -- Mamba-3 architecture in our minimal setting
        #                      (mathematically equivalent to complex by Prop 3)
        #   real_rope_frozen-- RoPE with frozen random rotation phases
        #                      (tests function-class contribution alone)
        configs = [
            ("real",              16),
            ("real",              128),
            ("complex",           16),
            ("real_rope",         16),
            ("real_rope_frozen",  16),
            ("real_rope_neg",     16),
        ]
        seeds = list(range(args.seeds))
        steps = args.steps

    plan = [(kind, n, s) for (kind, n) in configs for s in seeds]
    print(f"Total runs: {len(plan)}")
    print(f"K={args.K}  L={args.L}  A={A}  steps={steps}  lr={lr}  device={device}")

    t0 = time.time()
    rows = []
    for i, (kind, n_state, seed) in enumerate(plan, 1):
        tag = f"{kind}-n{n_state}-s{seed}"
        print(f"\n[{i}/{len(plan)}] {tag}")
        row = run_one(kind, n_state, K=args.K, L=args.L, A=A,
                       lr=lr, steps=steps, batch_size=batch_size,
                       device=device, seed=seed, tag=tag)
        rows.append(row)
    wallclock = time.time() - t0

    # Save JSON
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump({
            "config": args.config,
            "K": args.K, "L": args.L, "A": A,
            "steps": steps, "lr": lr, "batch_size": batch_size,
            "device": str(device),
            "rows": rows,
            "wallclock_sec": wallclock,
        }, f, indent=2, default=str)

    # Plots
    plot_comparison(rows, os.path.join(args.outdir, "loss_acc.png"),
                     title_suffix=f"  (K={args.K}, L={args.L}, A={A})")
    plot_param_dynamics(rows, os.path.join(args.outdir, "param_dynamics.png"))

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'kind':<10}{'n_state':>9}{'params':>10}{'seed':>6}"
          f"  {'final_acc':>10}{'final_loss':>12}")
    print("-" * 78)
    by = {}
    for r in rows:
        key = (r["kind"], r["n_state"])
        by.setdefault(key, []).append(r)
        print(f"{r['kind']:<10}{r['n_state']:>9}{r['params']:>10,}{r['seed']:>6}"
              f"  {r['final_acc']:>10.4f}{r['final_loss']:>12.4f}")
    print("-" * 78)
    print("Aggregated over seeds:")
    for (kind, n_state), lst in by.items():
        accs = np.array([r["final_acc"] for r in lst])
        print(f"  {kind:<10}n={n_state:<4d}  "
              f"acc {accs.mean():.4f} ± {accs.std()/math.sqrt(len(accs)):.4f}  "
              f"(n_seeds={len(lst)})")
    print(f"\nTotal wallclock: {wallclock/60:.1f} min")


if __name__ == "__main__":
    main()
