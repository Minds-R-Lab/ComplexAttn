"""
Training loop for the 3-class mod-3 task.

Structurally identical to train.py but uses cross-entropy + argmax
instead of BCE + sign threshold, and tracks the best-OOD checkpoint
(carrying over the lesson from train.py).
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_triple import make_dataset, make_depth_stratified_eval


def evaluate(model, tokens, labels, depths, device, batch_size=512):
    model.eval()
    correct, total = 0, 0
    per_d_c, per_d_n = {}, {}
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            t = tokens[i:i + batch_size].to(device)
            y = labels[i:i + batch_size].to(device)
            d = depths[i:i + batch_size]
            logits = model(t)
            pred = logits.argmax(dim=-1)
            ok = (pred == y).cpu()
            correct += ok.sum().item(); total += len(ok)
            for di, oki in zip(d.tolist(), ok.tolist()):
                per_d_c[di] = per_d_c.get(di, 0) + int(oki)
                per_d_n[di] = per_d_n.get(di, 0) + 1
    return correct / total, {d: per_d_c[d] / per_d_n[d] for d in sorted(per_d_n)}


def train_model(model, *,
                train_depth_range=(0, 5),
                eval_depths=tuple(range(0, 21)),
                n_train=60_000,
                n_eval_per_depth=2_000,
                batch_size=256,
                lr=5e-3,
                weight_decay=0.01,
                n_epochs=20,
                eval_every_steps=200,
                device="cuda",
                seed=0,
                tag="model"):
    torch.manual_seed(seed)

    train_tok, train_lab, _ = make_dataset(n_train, train_depth_range, seed=seed)
    eval_tok, eval_lab, eval_dep = make_depth_stratified_eval(
        n_eval_per_depth, list(eval_depths), seed=seed + 10_000)

    max_len = max(train_tok.size(1), eval_tok.size(1))
    if train_tok.size(1) < max_len:
        pad = torch.zeros(train_tok.size(0), max_len - train_tok.size(1), dtype=torch.long)
        train_tok = torch.cat([train_tok, pad], dim=1)
    if eval_tok.size(1) < max_len:
        pad = torch.zeros(eval_tok.size(0), max_len - eval_tok.size(1), dtype=torch.long)
        eval_tok = torch.cat([eval_tok, pad], dim=1)

    train_ds = TensorDataset(train_tok, train_lab)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_dl) * n_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    history = {"steps": [], "train_loss": [], "eval_acc": [], "by_depth": []}
    best_ood, best_step, best_state, best_by_depth = -1.0, 0, None, None

    print(f"\n[{tag}]  params={sum(p.numel() for p in model.parameters()):,}  "
          f"steps={total_steps}  device={device}")
    step, t0 = 0, time.time()
    for epoch in range(n_epochs):
        model.train()
        for tokens, y in train_dl:
            model.train()  # cuDNN-safe
            tokens = tokens.to(device); y = y.to(device).long()
            logits = model(tokens)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1

            if step % eval_every_steps == 0 or step == total_steps:
                acc, by_d = evaluate(model, eval_tok, eval_lab, eval_dep, device=device)
                history["steps"].append(step)
                history["train_loss"].append(loss.item())
                history["eval_acc"].append(acc)
                history["by_depth"].append(by_d)
                ind = [d for d in by_d if train_depth_range[0] <= d <= train_depth_range[1]]
                ood = [d for d in by_d if d > train_depth_range[1]]
                ind_acc = sum(by_d[d] for d in ind) / max(len(ind), 1)
                ood_acc = sum(by_d[d] for d in ood) / max(len(ood), 1)
                if ood_acc > best_ood:
                    best_ood, best_step = ood_acc, step
                    best_by_depth = dict(by_d)
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                print(f"  step {step:5d}/{total_steps}  loss={loss.item():.4f}  "
                      f"acc={acc:.3f}  ID={ind_acc:.3f}  OOD={ood_acc:.3f}")

    dt = time.time() - t0
    print(f"[{tag}]  done in {dt:.1f}s  (best OOD={best_ood:.3f} @ step {best_step})")

    if best_state is not None:
        model.load_state_dict(best_state)
    final_acc, final_by_depth = evaluate(model, eval_tok, eval_lab, eval_dep, device=device)
    history["final_acc"]      = final_acc
    history["final_by_depth"] = final_by_depth
    history["best_step"]      = best_step
    history["best_ood"]       = best_ood
    history["wallclock_sec"]  = dt
    return model, history
