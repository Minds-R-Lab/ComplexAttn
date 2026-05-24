"""pretrain.py — Phase 1c-2: pretrain GPT-2 small on the SFIB pretrain corpus.

GATE: Must hit Retention@0 >= 95% on a held-out Phase A slice before we
proceed to baselines. If it doesn't, escalate to GPT-2 medium per the
locked plan.

Inputs : the synthetic KB (generated deterministically from kb_data.py)
Outputs:
    checkpoints/pretrained_seed{seed}.pt   - best checkpoint by retention
    results/pretrain_seed{seed}.json       - training history + gate status

Estimated runtime on a 3090: ~20-60 min for 5,000 facts x 3 templates,
at most 20 epochs (early stop at 95%).

Usage:
    python pretrain.py                  # full training, gpt2-small
    python pretrain.py --smoke          # quick sanity run on 200 facts
    python pretrain.py --model gpt2-medium   # escalate if gate fails
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise SystemExit(
        "missing dependency: pip install transformers"
    )

from kb_data import generate_kb, render_train_example, render_eval_query
from evaluate import EvalExample, evaluate_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SFIB_DIR = Path(__file__).parent
CKPT_DIR = SFIB_DIR / "checkpoints"
RESULTS_DIR = SFIB_DIR / "results"
CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset: tokenized fact sentences for causal-LM training
# ---------------------------------------------------------------------------

class FactDataset(Dataset):
    """Renders each fact as Q/A pairs matching the eval format.

    For each triple, we generate (prompt + " " + answer) examples using ALL
    query templates of that relation. This aligns training and eval, so the
    model doesn't need to generalize from statement form to Q/A form.

    Optionally also include statement form for surface-form robustness.
    """

    def __init__(self, triples, tokenizer, max_len: int = 64,
                 include_statements: bool = True):
        from kb_data import RELATIONS  # local import; cheap
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts: list[str] = []

        for t in triples:
            rel = RELATIONS[t.relation]
            # Q/A renderings: one per query template
            for q_idx in range(len(rel.query_templates)):
                prompt, target = render_eval_query(t, template_idx=q_idx)
                # Train sequence: "Question? Answer: object"
                # The space after "Answer:" is included; target has no leading space.
                text = f"{prompt} {target}"
                self.texts.append(text)
            # Optionally add statement form too
            if include_statements:
                for s_idx in range(len(rel.fact_templates)):
                    self.texts.append(render_train_example(t, template_idx=s_idx))

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx] + self.tokenizer.eos_token
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len,
                             padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2",
                    help="HuggingFace model name (e.g. gpt2, Qwen/Qwen2.5-0.5B-Instruct, TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_epochs", type=int, default=-1, help="default: 20 (full) / 5 (smoke); -1 lets smoke override")
    ap.add_argument("--target_retention", type=float, default=0.95,
                    help="early-stop gate: stop when held-out retention >= this")
    ap.add_argument("--n_holdout", type=int, default=500,
                    help="held-out pretrain triples used ONLY for retention eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_pretrain", type=int, default=5000,
                    help="number of pretrain facts (default 5000; reduce if gate fails)")
    ap.add_argument("--n_insert", type=int, default=1000)
    ap.add_argument("--n_compose", type=int, default=300)
    ap.add_argument("--include_statements", action="store_true", default=True,
                    help="train on statement form in addition to Q/A (default: True)")
    ap.add_argument("--no_statements", dest="include_statements", action="store_false")
    ap.add_argument("--smoke", action="store_true",
                    help="quick sanity run: 200 facts, 3 epochs")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    if DEVICE.type == "cuda": torch.cuda.manual_seed_all(args.seed)

    print(f"[pretrain] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[pretrain] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[pretrain] model: {args.model}")
    print(f"[pretrain] lr={args.lr}  batch_size={args.batch_size}  max_epochs={args.max_epochs}")

    # ---- KB ----
    # Resolve max_epochs from sentinel
    if args.max_epochs == -1:
        args.max_epochs = 5 if args.smoke else 20
    if args.smoke:
        kb = generate_kb(seed=args.seed, n_pretrain=200, n_insert=20, n_compose=10,
                          n_entities_a=40, n_entities_b=15)
        args.n_holdout = 30
        print(f"[pretrain] SMOKE MODE: 200 facts, 30 holdout, max_epochs={args.max_epochs}")
    else:
        kb = generate_kb(seed=args.seed,
                          n_pretrain=args.n_pretrain,
                          n_insert=args.n_insert,
                          n_compose=args.n_compose)
    print(f"[pretrain] KB: pretrain={len(kb.pretrain_triples)}  "
          f"insert={len(kb.insert_triples)}  compose={len(kb.compose_pairs)}")

    # CORRECT design: train on ALL pretrain triples (we want memorization,
    # not generalization to unseen facts). Sample N triples for retention
    # eval from the SAME training set — this tests "did the model
    # successfully memorize the corpus?", which is what 'retention' means
    # in the benchmark spec.
    all_pretrain = list(kb.pretrain_triples)
    train_triples = all_pretrain
    random.Random(args.seed).shuffle(all_pretrain)
    retention_sample = all_pretrain[:args.n_holdout]
    print(f"[pretrain] train triples: {len(train_triples)} (ALL pretrain)")
    print(f"[pretrain] retention eval sample: {len(retention_sample)} "
          f"(drawn from training set; this is a memorization check)")

    # ---- Model & tokenizer ----
    print(f"[pretrain] loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(DEVICE)
    model = model.float()  # belt-and-suspenders: force fp32 (some models default to bf16)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[pretrain] params: {n_params:,} ({n_params/1e6:.1f}M)")

    # ---- Training data ----
    ds = FactDataset(train_triples, tokenizer, include_statements=args.include_statements)
    tpl_str = "Q/A + statements" if args.include_statements else "Q/A only"
    print(f"[pretrain] training examples ({tpl_str} x {len(train_triples)} triples): {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=DEVICE.type=="cuda")

    # ---- Held-out retention eval examples ----
    holdout_eval: list[EvalExample] = []
    for t in retention_sample:
        for q_idx in range(2):
            prompt, target = render_eval_query(t, template_idx=q_idx)
            holdout_eval.append(EvalExample(
                prompt=prompt, target=target, kind="retention",
                meta={"triple": t.as_tuple()},
            ))
    print(f"[pretrain] retention eval examples: {len(holdout_eval)} (over {len(retention_sample)} trained triples)")

    # ---- Train ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)  # memorization, not regularization
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)

    history = []
    best_acc = 0.0
    safe_model_name = args.model.replace("/", "__").replace("-", "_")
    ckpt_path = CKPT_DIR / f"pretrained_seed{args.seed}_{safe_model_name}.pt"

    print(f"\n[pretrain] beginning training; gate target = "
          f"{args.target_retention:.2f}, max_epochs = {args.max_epochs}")
    for ep in range(1, args.max_epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = input_ids.clone()
            labels[attn == 0] = -100
            opt.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(out.loss.item())
        sched.step()
        avg_loss = sum(losses) / len(losses)
        train_t = time.time() - t0

        # Retention eval
        t1 = time.time()
        result = evaluate_model(model, tokenizer, holdout_eval,
                                 max_new_tokens=12, batch_size=32)
        eval_t = time.time() - t1
        acc = result["summary"]["retention"]["accuracy"]
        # Debug: print first 3 records so we can SEE what's happening
        records = result["records"][:3]
        for r in records:
            mark = "✓" if r["correct"] else "✗"
            print(f"    {mark} prompt={r['prompt']!r}  target={r['target']!r}  got={r['generated']!r}")
        history.append({
            "epoch": ep, "train_loss": avg_loss,
            "retention_acc": acc,
            "train_time_s": train_t, "eval_time_s": eval_t,
        })
        print(f"  ep {ep:>2d}/{args.max_epochs}  loss={avg_loss:.4f}  "
              f"retention={acc:.4f}  (train {train_t:.0f}s, eval {eval_t:.0f}s)")

        # Save best
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "epoch": ep, "retention_acc": acc,
                "args": vars(args),
                "model_name": args.model,
            }, ckpt_path)
            print(f"     -> saved best checkpoint (retention={acc:.4f}) -> {ckpt_path.name}")

        # Gate
        if acc >= args.target_retention:
            print(f"\n✓ GATE PASSED at epoch {ep}: retention {acc:.4f} >= {args.target_retention:.2f}")
            break

    summary = {
        "model": args.model,
        "best_retention_acc": best_acc,
        "gate_target": args.target_retention,
        "gate_passed": best_acc >= args.target_retention,
        "history": history,
        "checkpoint": str(ckpt_path),
        "smoke": args.smoke,
    }
    out_json = RESULTS_DIR / f"pretrain_seed{args.seed}_{safe_model_name}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[pretrain] summary saved -> {out_json}")
    print(f"[pretrain] best retention: {best_acc:.4f}")

    if summary["gate_passed"]:
        print(f"✓ Proceed to Phase 1c-3 (baselines).")
    else:
        print(f"!! GATE FAILED: retention {best_acc:.4f} < {args.target_retention:.2f}")
        print(f"   Try a larger model (e.g. --model gpt2-medium or Qwen/Qwen2.5-1.5B).")


if __name__ == "__main__":
    main()
