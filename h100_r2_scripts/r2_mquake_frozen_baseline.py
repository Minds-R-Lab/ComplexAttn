"""r2_mquake_frozen_baseline.py -- Reviewer #1 R1.2 (R2 revision).

Frozen-base counterpart of r1_2_mquake_eval.py: evaluates the UNEDITED
Mamba on MQuAKE single-hop and multi-hop questions, with no slot-bank
insertion. Fills the '[pending H100 cell]' in manuscript Table 8
(tab:mquake) under Section 6.5.

Shares the data loader, item format, scoring function, and output JSON
schema with r1_2_mquake_eval.py so downstream aggregation code (and the
manuscript's eventual numerical insert) treats the two cells identically.

Usage (run from sfib_mamba/ directory, same as its sibling):
    cd sfib_mamba/
    python r2_mquake_frozen_baseline.py \\
        --model state-spaces/mamba-790m-hf --layer 32 \\
        --n_items 500 --seed 0 \\
        --out results/mquake_frozen_mamba790m_seed0.json

Expected wall time at Mamba-790M on 500 items: ~45-60 min (no editor
inserts, just greedy generation on the questions).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def load_mquake(n_items: int, seed: int):
    """Same loader as r1_2_mquake_eval.load_mquake."""
    import random
    JSON_PATH = Path(__file__).parent.parent / "mquake_repo" / "datasets" / "MQuAKE-CF-3k.json"
    with open(JSON_PATH) as fh:
        items = json.load(fh)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n_items]


def score_multihop(generated: str, correct_answers: list[str]) -> bool:
    g = generated.strip().lower()
    return any(a.strip().lower() in g for a in correct_answers if a)


def _tn_str(tn) -> str:
    return tn["str"] if isinstance(tn, dict) else tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba-790m-hf")
    ap.add_argument("--layer", type=int, default=32,
                    help="Echoed into the output JSON for schema parity; not used.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[mquake-frozen] device: {DEVICE}  model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    items = load_mquake(args.n_items, args.seed)
    print(f"[mquake-frozen] loaded {len(items)} MQuAKE items")

    per_item = []
    hits_singlehop = 0
    hits_multihop = 0
    total_singlehop = 0
    total_multihop = 0
    t_total = time.time()

    for i, item in enumerate(items, start=1):
        # Single-hop: for each requested rewrite, probe its prompt on the
        # UNEDITED model (the frozen base cannot know the new fact; this
        # measures prior-based single-hop match).
        sh_hits = 0
        for rr in item["requested_rewrite"]:
            prompt_text = rr["prompt"].format(rr["subject"])
            target_new = _tn_str(rr["target_new"])
            ids = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                gen = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id)
            txt = tokenizer.decode(gen[0, ids["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            if score_multihop(txt, [target_new]):
                sh_hits += 1
        total_singlehop += len(item["requested_rewrite"])
        hits_singlehop += sh_hits

        # Multi-hop: probe each composed question on the UNEDITED model.
        correct = [item.get("new_answer", "")] + item.get("new_answer_alias", [])
        correct = [c for c in correct if c]
        mh_hits = 0
        for q in item["questions"]:
            ids = tokenizer(q, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                gen = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id)
            txt = tokenizer.decode(gen[0, ids["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            if score_multihop(txt, correct):
                mh_hits += 1
        total_multihop += len(item["questions"])
        hits_multihop += mh_hits

        per_item.append({
            "case_id": item.get("case_id"),
            "n_edits": len(item["requested_rewrite"]),
            "singlehop_hits": sh_hits,
            "singlehop_total": len(item["requested_rewrite"]),
            "multihop_hits": mh_hits,
            "multihop_total": len(item["questions"]),
        })

        if i % 25 == 0:
            print(f"[mquake-frozen] {i}/{len(items)}  "
                  f"single-hop acc={hits_singlehop/max(1,total_singlehop):.4f}  "
                  f"multi-hop acc={hits_multihop/max(1,total_multihop):.4f}  "
                  f"({(time.time()-t_total)/60:.1f} min)")
            with open(args.out, "w") as fh:
                json.dump({
                    "method": "frozen",
                    "benchmark": "mquake-cf-3k",
                    "model": args.model, "layer": args.layer, "seed": args.seed,
                    "n_items": args.n_items,
                    "singlehop_accuracy": hits_singlehop / max(1, total_singlehop),
                    "multihop_accuracy":  hits_multihop  / max(1, total_multihop),
                    "n_items_processed": i,
                    "wall_s": time.time() - t_total,
                    "per_item": per_item,
                }, fh, indent=2)

    with open(args.out, "w") as fh:
        json.dump({
            "method": "frozen",
            "benchmark": "mquake-cf-3k",
            "model": args.model, "layer": args.layer, "seed": args.seed,
            "n_items": args.n_items,
            "singlehop_accuracy": hits_singlehop / max(1, total_singlehop),
            "multihop_accuracy":  hits_multihop  / max(1, total_multihop),
            "n_items_processed": len(items),
            "wall_s": time.time() - t_total,
            "per_item": per_item,
        }, fh, indent=2)
    print(f"[mquake-frozen] final  single-hop acc={hits_singlehop/max(1,total_singlehop):.4f}  "
          f"multi-hop acc={hits_multihop/max(1,total_multihop):.4f}  "
          f"wall={(time.time()-t_total)/3600:.2f} h")
    print(f"[mquake-frozen] wrote {args.out}")


if __name__ == "__main__":
    main()
