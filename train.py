"""
Training and evaluation for the parity-of-negation experiment.

Trains a model with AdamW + cosine schedule, evaluates accuracy stratified
by negation depth, and returns a complete metrics dictionary so the
orchestrator can plot everything later.
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import make_dataset, make_depth_stratified_eval


def evaluate(model, tokens, labels, depths, device, batch_size=512):
    """Return (overall_acc, dict_of_per_depth_acc)."""
    model.eval()
    correct_total = 0
    total = 0
    per_depth_correct = {}
    per_depth_total   = {}

    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            t = tokens[i:i + batch_size].to(device)
            y = labels[i:i + batch_size].to(device)
            d = depths[i:i + batch_size]
            logits = model(t)
            pred = (logits > 0).long()
            ok = (pred == y).cpu()
            correct_total += ok.sum().item()
            total += len(ok)
            for di, oki in zip(d.tolist(), ok.tolist()):
                per_depth_correct[di] = per_depth_correct.get(di, 0) + int(oki)
                per_depth_total[di]   = per_depth_total.get(di, 0) + 1

    overall = correct_total / total
    by_depth = {d: per_depth_correct[d] / per_depth_total[d]
                for d in sorted(per_depth_total)}
    return overall, by_depth


def train_model(model, *,
                train_depth_range=(0, 3),
                eval_depths=tuple(range(0, 11)),
                n_train=50_000,
                n_eval_per_depth=1000,
                batch_size=256,
                lr=3e-4,
                weight_decay=0.01,
                n_epochs=20,
                eval_every_steps=200,
                device="cuda",
                seed=0,
                tag="model"):
    """
    Train one model and return a metrics dict.

    train_depth_range : (min_nots, max_nots) seen during training
    eval_depths       : depths to evaluate on (in & out of distribution)
    """
    torch.manual_seed(seed)

    # Data
    train_tok, train_lab, _ = make_dataset(n_train, train_depth_range,
                                           seed=seed)
    eval_tok,  eval_lab, eval_dep = make_depth_stratified_eval(
        n_eval_per_depth, list(eval_depths), seed=seed + 10_000)

    # Pad eval to at least match train max-length so positions are seen.
    max_len = max(train_tok.size(1), eval_tok.size(1))
    if train_tok.size(1) < max_len:
        pad = torch.zeros(train_tok.size(0), max_len - train_tok.size(1),
                          dtype=torch.long)
        train_tok = torch.cat([train_tok, pad], dim=1)
    if eval_tok.size(1) < max_len:
        pad = torch.zeros(eval_tok.size(0), max_len - eval_tok.size(1),
                          dtype=torch.long)
        eval_tok = torch.cat([eval_tok, pad], dim=1)

    train_ds = TensorDataset(train_tok, train_lab)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=True)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    total_steps = len(train_dl) * n_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    history = {
        "steps":     [],
        "train_loss":[],
        "eval_acc":  [],
        "by_depth":  [],
    }

    print(f"\n[{tag}]  params={sum(p.numel() for p in model.parameters()):,}  "
          f"steps={total_steps}  device={device}")
    step = 0
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        for tokens, y in train_dl:
            tokens = tokens.to(device)
            y      = y.to(device).float()
            logits = model(tokens)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % eval_every_steps == 0 or step == total_steps:
                acc, by_d = evaluate(model, eval_tok, eval_lab, eval_dep,
                                     device=device)
                history["steps"].append(step)
                history["train_loss"].append(loss.item())
                history["eval_acc"].append(acc)
                history["by_depth"].append(by_d)
                ind = [d for d in by_d if train_depth_range[0] <= d <= train_depth_range[1]]
                ood = [d for d in by_d if d > train_depth_range[1]]
                ind_acc = sum(by_d[d] for d in ind) / max(len(ind), 1)
                ood_acc = sum(by_d[d] for d in ood) / max(len(ood), 1)
                print(f"  step {step:5d}/{total_steps}  "
                      f"loss={loss.item():.4f}  "
                      f"acc={acc:.3f}  "
                      f"ID={ind_acc:.3f}  OOD={ood_acc:.3f}")

    dt = time.time() - t0
    print(f"[{tag}]  done in {dt:.1f}s")

    # Final detailed eval
    final_acc, final_by_depth = evaluate(model, eval_tok, eval_lab, eval_dep,
                                         device=device)
    history["final_acc"] = final_acc
    history["final_by_depth"] = final_by_depth
    history["wallclock_sec"] = dt
    return model, history
