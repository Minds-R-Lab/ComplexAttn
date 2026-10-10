"""r1_2_mquake_eval.py -- Reviewer #1 point 2 (ESWA R1.2).

Reviewer #1 asks for a multi-hop evaluation beyond CounterFact and
zsRE's single-hop format. MQuAKE (Zhong et al. 2023, EMNLP) is the
standard multi-hop knowledge-editing benchmark: each item specifies
a chain of 2-4 single-hop facts and a composed multi-hop question
whose answer depends on all of them.

Protocol
--------
We follow the MQuAKE-CF-3K subset (3,000 items, 1-4 edits per item,
2-4 hops per question).

For each item:
    1. Insert all single-hop edits into STEER's slot bank sequentially.
    2. Evaluate the composed multi-hop question with greedy generation.
    3. Score as a substring match against any of the ground-truth
       answers (MQuAKE allows multiple valid phrasings).

Metrics
-------
    Edit-wise accuracy  (equivalent to Eff on single-hop facts)
    Multi-hop accuracy  (new; what the reviewer wants)
    Multi-hop subject-of-interest consistency

This script does NOT re-use the CounterFact / zsRE eval harness because
multi-hop scoring is different. We fetch MQuAKE from HuggingFace Hub
(dataset id: 'yuan-tian/MQuAKE').

Usage:
    cd sfib_mamba/
    python r1_2_mquake_eval.py \\
        --model state-spaces/mamba-790m-hf --layer 32 \\
        --n_items 500 --seed 0 \\
        --out results/mquake_steer_mamba790m_seed0.json

Expected wall time at Mamba-790M, 500 items (avg 2.5 edits each -> 1250
insertions): ~3-4 h.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import shard_mamba  # noqa: F401
import shard_mamba_realdata_patches  # noqa: F401
from shard_mamba import SHARDMambaMethod

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def load_mquake(n_items: int, seed: int):
    """Fetch MQuAKE-CF-3K and return a shuffled slice of `n_items`.

    MQuAKE items have the structure:
        {
          "case_id": int,
          "requested_rewrite": [ { "prompt": str, "subject": str, "target_new": str, "target_true": str }, ... ],
          "questions":  [ str, ... ],
          "new_answer": str,
          "new_answer_alias": [ str, ... ],
          "answer":     str,
          "answer_alias": [ str, ... ],
        }
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("yuan-tian/MQuAKE", "cf-3k", split="train")
    except Exception as e:
        raise RuntimeError(
            f"MQuAKE download failed: {e!r}\n"
            "Alternative: clone the MQuAKE repo "
            "(https://github.com/princeton-nlp/MQuAKE) and point this loader "
            "at datasets/MQuAKE-CF-3k.json."
        )
    import random
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    return [ds[i] for i in idx[:n_items]]


def score_multihop(generated: str, correct_answers: list[str]) -> bool:
    """Substring match against any of the gold-answer phrasings."""
    g = generated.strip().lower()
    return any(a.strip().lower() in g for a in correct_answers if a)


class EditTriple:
    """Minimal shim so SHARDMambaMethod.insert() accepts an MQuAKE
    'requested_rewrite' dict as if it were a CounterFact record."""
    def __init__(self, rr):
        self.prompt_template = rr["prompt"]
        self.subject = rr["subject"]
        self.target_new = rr["target_new"]
        self.target_true = rr.get("target_true", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba-790m-hf")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--v_steps", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[mquake] device: {DEVICE}  model: {args.model}  layer: {args.layer}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    items = load_mquake(args.n_items, args.seed)
    print(f"[mquake] loaded {len(items)} MQuAKE items")

    method = SHARDMambaMethod(
        layer_idx=args.layer, kind="out_proj",
        sim_threshold=args.tau,
        v_steps=args.v_steps, v_lr=1.0,
        v_weight_decay=0.0, v_norm_constraint=20.0,
        capture_position="prompt_last", fire_position="last",
    )
    method.setup(model, tokenizer, kb=None)

    per_item = []
    hits_singlehop = 0
    hits_multihop = 0
    total_singlehop = 0
    total_multihop = 0
    t_total = time.time()

    for i, item in enumerate(items, start=1):
        # Insert all requested rewrites for this item
        for rr in item["requested_rewrite"]:
            method.insert(EditTriple(rr))

        # Single-hop: for each inserted edit, probe its own prompt
        sh_hits = 0
        for rr in item["requested_rewrite"]:
            ids = tokenizer(rr["prompt"], return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                gen = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id)
            txt = tokenizer.decode(gen[0, ids["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            if score_multihop(txt, [rr["target_new"]]):
                sh_hits += 1
        total_singlehop += len(item["requested_rewrite"])
        hits_singlehop += sh_hits

        # Multi-hop
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
            print(f"[mquake] {i}/{len(items)}  "
                  f"single-hop acc={hits_singlehop/max(1,total_singlehop):.4f}  "
                  f"multi-hop acc={hits_multihop/max(1,total_multihop):.4f}  "
                  f"({(time.time()-t_total)/60:.1f} min)")
            # Persist checkpoint
            with open(args.out, "w") as fh:
                json.dump({
                    "method": "steer",
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
            "method": "steer",
            "benchmark": "mquake-cf-3k",
            "model": args.model, "layer": args.layer, "seed": args.seed,
            "n_items": args.n_items,
            "singlehop_accuracy": hits_singlehop / max(1, total_singlehop),
            "multihop_accuracy":  hits_multihop  / max(1, total_multihop),
            "n_items_processed": len(items),
            "wall_s": time.time() - t_total,
            "per_item": per_item,
        }, fh, indent=2)
    print(f"[mquake] final  single-hop acc={hits_singlehop/max(1,total_singlehop):.4f}  "
          f"multi-hop acc={hits_multihop/max(1,total_multihop):.4f}  "
          f"wall={(time.time()-t_total)/3600:.2f} h")
    print(f"[mquake] wrote {args.out}")


if __name__ == "__main__":
    main()
