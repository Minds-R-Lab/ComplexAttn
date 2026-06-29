"""r2_composition_oracle.py -- Reviewer #2 point R2.3 in-context oracle.

For each Qwen2.5 scale {0.5B, 1.5B, 3B, 7B}, we test in-context
composition with the relevant facts provided directly in the prompt.
This isolates the bottleneck:

  - If oracle accuracy is high, the backbone CAN compose two facts when
    they're given to it. The SFIB Com@500 ~ 0 result is then a
    retrieval/representation problem at the editor level, not a
    backbone capability problem.

  - If oracle accuracy is also low, the backbone cannot reliably
    perform two-hop reasoning at this scale, and SHARD's Com@500 ~ 0
    is exactly what the model can support given perfect retrieval.

We build synthetic SFIB-style fact pairs directly via kb_data.generate_kb
(no fine-tuning required: the oracle prompts the model with both facts
each time) and ask the two-hop question with both facts visible.

Usage on the H100 box:
    cd sfib/
    python r2_composition_oracle.py --models \\
        Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \\
        Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct \\
        --out results/composition_oracle.json

Expected wall time on a single H100-80GB: about 4 hr across all four scales.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kb_data import generate_kb, render_composition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SFIB_DIR = Path(__file__).parent
RESULTS_DIR = SFIB_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]


def _statement_of(triple, template_idx: int = 0) -> str:
    from kb_data import render_train_example
    return render_train_example(triple, template_idx=template_idx)


def build_oracle_prompts(kb, n_inserted: int):
    """For each composition pair whose inserted triple has index < n_inserted,
    construct a single prompt that prepends BOTH the inserted fact and the
    pretrain fact as statement-form context, followed by the two-hop question."""
    insert_set = {t.as_tuple() for t in kb.insert_triples[:n_inserted]}
    out = []
    for (t_pre, t_ins, qtext) in kb.compose_pairs:
        if t_ins.as_tuple() not in insert_set:
            continue
        prompt_q, target = render_composition(t_pre, t_ins, qtext)
        fact_a = _statement_of(t_ins)
        fact_b = _statement_of(t_pre)
        oracle_prompt = f"{fact_a} {fact_b} {prompt_q}"
        out.append({
            "prompt": oracle_prompt,
            "target": target,
            "meta": {
                "inserted_triple": t_ins.as_tuple(),
                "pretrain_triple": t_pre.as_tuple(),
                "question_text": qtext,
            },
        })
    return out


def evaluate_model(model_name, oracle_examples, max_new_tokens):
    print(f"\n== {model_name}: oracle composition on {len(oracle_examples)} pairs ==")
    if DEVICE.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.float16 if DEVICE.type == "cuda" else torch.float32),
        device_map="auto",
    )
    model.eval()

    correct = 0
    per_example = []
    t0 = time.time()
    for i, ex in enumerate(oracle_examples, start=1):
        ids = tokenizer(ex["prompt"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(
                **ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[0, ids["input_ids"].shape[1]:]
        out_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        target = ex["target"].strip()
        is_ok = target.lower() in out_text.lower()
        per_example.append({
            "prompt": ex["prompt"], "target": target, "generated": out_text,
            "correct": bool(is_ok), "meta": ex["meta"],
        })
        correct += int(is_ok)
        if i % 50 == 0:
            print(f"   {i}/{len(oracle_examples)}  running acc = {correct/i:.3f}")
    wall = time.time() - t0

    del model
    torch.cuda.empty_cache()
    return {
        "model": model_name,
        "n": len(oracle_examples),
        "oracle_acc": correct / max(1, len(oracle_examples)),
        "wall_s": wall,
        "per_example": per_example,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_pretrain", type=int, default=2000)
    ap.add_argument("--n_insert", type=int, default=500)
    ap.add_argument("--n_compose", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[oracle] device: {DEVICE}")
    print(f"[oracle] models: {args.models}")

    kb = generate_kb(seed=args.seed,
                     n_pretrain=args.n_pretrain,
                     n_insert=args.n_insert,
                     n_compose=args.n_compose)
    print(f"[oracle] KB: pretrain={len(kb.pretrain_triples)}  "
          f"insert={len(kb.insert_triples)}  compose={len(kb.compose_pairs)}")
    oracle_examples = build_oracle_prompts(kb, n_inserted=args.n_insert)
    print(f"[oracle] {len(oracle_examples)} oracle composition prompts built")

    per_model = []
    for m in args.models:
        row = evaluate_model(m, oracle_examples, args.max_new_tokens)
        per_model.append(row)
        with open(args.out, "w") as f:
            json.dump({"per_model": per_model}, f, indent=2)
        print(f"[oracle] {m}  oracle_acc = {row['oracle_acc']:.3f}  "
              f"[{row['wall_s']:.0f}s]")

    print(f"\n[oracle] wrote {args.out}")
    print("Summary:")
    for row in per_model:
        print(f"   {row['model']:>40}  oracle_acc = {row['oracle_acc']:.3f}")


if __name__ == "__main__":
    main()
