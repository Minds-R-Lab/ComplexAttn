"""diagnose_in_context.py — does GPT-2 small respond to ANY prompt format that
makes the fact accessible at inference time?

Tries 5 prompt formats on retention examples (which we know the model has
memorized). If accuracy stays low even with the fact in the prompt, the
problem is model scale (no in-context-learning at 124M). If one format
recovers accuracy to ~95%, the problem is just format choice — we update the
in_context method to use that format.

Usage:
    python diagnose_in_context.py --ckpt checkpoints/pretrained_seed0_gpt2.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from kb_data import generate_kb, render_train_example, render_eval_query
from evaluate import EvalExample, evaluate_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pretrained_seed0_gpt2.pt")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_pretrain", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=200,
                    help="how many retention triples to test (2 queries each)")
    args = ap.parse_args()

    print(f"[diag-ICL] device: {DEVICE}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    kb = generate_kb(seed=args.seed, n_pretrain=args.n_pretrain,
                      n_insert=500, n_compose=200)
    all_p = list(kb.pretrain_triples)
    random.Random(args.seed).shuffle(all_p)
    test = all_p[:args.n_test]
    print(f"[diag-ICL] testing {len(test)} retention triples, 2 query templates each = {2*len(test)} examples")

    # Build the eval examples once per format
    formats = {
        # 0: no prepend, baseline
        "no_prepend":
            lambda fact_stmt, q: q,
        # 1: simple prepend with space (CURRENT in_context implementation)
        "stmt_space":
            lambda fact_stmt, q: f"{fact_stmt} {q}",
        # 2: simple prepend with newline (sentence boundary)
        "stmt_newline":
            lambda fact_stmt, q: f"{fact_stmt}\n{q}",
        # 3: explicit "Fact:" tag, mimics structured prompt
        "fact_tag":
            lambda fact_stmt, q: f"Fact: {fact_stmt}\nQuestion: {q}",
        # 4: few-shot — one in-domain Q/A example, then the test Q
        # Use a fixed exemplar so it's stable across runs
        "few_shot":
            lambda fact_stmt, q: (
                f"{fact_stmt}\n"
                f"Q: What is being described in the previous sentence? "
                f"Answer: the fact above.\n"
                f"{q}"
            ),
        # 5: question-then-fact form (sometimes works on small LMs)
        "q_then_fact":
            lambda fact_stmt, q: f"{q[:-len('Answer:')].strip()} ({fact_stmt}) Answer:",
    }

    # Run each format
    results = {}
    for fmt_name, fmt_fn in formats.items():
        examples: list[EvalExample] = []
        for t in test:
            fact_stmt = render_train_example(t, template_idx=0)
            for q_idx in range(2):
                q, a = render_eval_query(t, template_idx=q_idx)
                prompt = fmt_fn(fact_stmt, q)
                examples.append(EvalExample(
                    prompt=prompt, target=a, kind="retention",
                    meta={"triple": t.as_tuple(), "q_idx": q_idx},
                ))
        result = evaluate_model(model, tokenizer, examples,
                                 max_new_tokens=12, batch_size=32)
        acc = result["summary"]["retention"]["accuracy"]
        results[fmt_name] = acc
        print(f"  {fmt_name:<14}  {acc:.4f}")
        # Print 2 sample outputs per format
        for r in result["records"][:2]:
            mark = "✓" if r["correct"] else "✗"
            short_prompt = r["prompt"].replace("\n", " ⏎ ")
            if len(short_prompt) > 110:
                short_prompt = short_prompt[:107] + "..."
            print(f"    {mark} {short_prompt}")
            print(f"       target='{r['target']}'  got='{r['generated'].strip()[:50]}'")

    print()
    print("== summary ==")
    for k, v in results.items():
        print(f"  {k:<14}  {v:.4f}")
    best = max(results, key=results.get)
    print(f"  best format: {best} = {results[best]:.4f}")


if __name__ == "__main__":
    main()
