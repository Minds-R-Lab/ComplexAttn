"""run_grace.py -- standalone GRACE runner for CounterFact and zsRE.

Reuses cf_eval (from run_counterfact.py) and zsre_eval (from run_zsre.py)
verbatim so the metrics are identical to the existing baseline numbers.
The only thing this script adds is the GRACE insertion loop.

Usage:
    python run_grace.py --benchmark counterfact --model Qwen/Qwen2.5-0.5B-Instruct \
        --grace_layer 17 --grace_n_steps 100 --grace_lr 1.0 --grace_eps_init 1.0 \
        --seed 0 --n_edits 500 \
        --out results/counterfact_grace_qwen0_5b_seed0.json

    python run_grace.py --benchmark zsre --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --grace_layer 13 --grace_n_steps 100 --grace_lr 1.0 --grace_eps_init 1.0 \
        --seed 0 --n_edits 500 \
        --out results/zsre_grace_tinyllama_seed0.json

Default layers (mid-to-late MLP, matching the SHARD/AddressableMemory runs):
    GPT-2 small  (12 layers)        ->  5  (~42%, MEMIT default)
    Qwen2.5-0.5B (24 layers)        -> 17  (~71%)
    Qwen2.5-1.5B/7B (28 layers)     -> 20  (~71%)
    Qwen2.5-3B   (36 layers)        -> 26  (~72%)
    TinyLlama-1.1B (22 layers)      -> 13  (~59%)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import grace_method                              # noqa: F401
import grace_realdata_patches                    # noqa: F401
from grace_method import GRACEMethod

SFIB_DIR = Path(__file__).parent
RESULTS_DIR = SFIB_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_benchmark_and_eval(benchmark: str, seed: int, n_edits: int):
    if benchmark == "counterfact":
        from counterfact_data import cf_splits
        from run_counterfact import cf_eval
        edits, holdout = cf_splits(n_edits=n_edits, seed=seed)
        return edits, holdout, cf_eval
    if benchmark == "zsre":
        from zsre_data import zsre_splits
        from run_zsre import zsre_eval
        edits, holdout = zsre_splits(n_edits=n_edits, seed=seed)
        return edits, holdout, zsre_eval
    raise ValueError(f"Unknown benchmark: {benchmark!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["counterfact", "zsre"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--eval_at", default="0,1,10,50,100,250,500")
    ap.add_argument("--batch_size_eval", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=12)

    ap.add_argument("--grace_layer", type=int, default=17)
    ap.add_argument("--grace_n_steps", type=int, default=100)
    ap.add_argument("--grace_lr", type=float, default=1.0)
    ap.add_argument("--grace_eps_init", type=float, default=1.0)
    ap.add_argument("--grace_init", choices=["zeros", "random"], default="zeros")

    ap.add_argument("--out", required=True)

    args = ap.parse_args()

    print(f"[grace] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[grace] GPU: {torch.cuda.get_device_name()}")
    print(f"[grace] benchmark: {args.benchmark}")
    print(f"[grace] model: {args.model}")
    print(f"[grace] {args.n_edits} edits, eval_at = {args.eval_at}")

    edits, holdout, eval_fn = load_benchmark_and_eval(
        args.benchmark, args.seed, args.n_edits,
    )
    print(f"[grace] loaded {len(edits)} edits, {len(holdout)} held-out specificity")

    eval_at = [int(x) for x in args.eval_at.split(",")]

    print(f"[grace] loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    method = GRACEMethod(
        layer_idx=args.grace_layer,
        n_steps=args.grace_n_steps,
        lr=args.grace_lr,
        eps_init=args.grace_eps_init,
        init=args.grace_init,
    )
    method.setup(model, tokenizer, kb=None)

    results: list = []

    def eval_and_log(N: int, tag: str = "") -> None:
        t0 = time.time()
        result = eval_fn(
            model, tokenizer,
            edits_seen=edits[:N],
            holdout=holdout,
            batch_size=args.batch_size_eval,
            max_new_tokens=args.max_new_tokens,
        )
        eval_t = time.time() - t0
        eff = result["efficacy"]
        gen = result["generalization"]
        spec = result["specificity"]

        def fmt(x):
            return "n/a   " if x is None else f"{x:.4f}"

        print(f"  N={N:<5} "
              f"Eff={fmt(eff['accuracy'])} (n={eff['n']})   "
              f"Gen={fmt(gen['accuracy'])} (n={gen['n']})   "
              f"Spec={fmt(spec['accuracy'])} (n={spec['n']})   "
              f"[eval {eval_t:.1f}s]")
        results.append({
            "N": N,
            "efficacy": eff,
            "generalization": gen,
            "specificity": spec,
            "n_slots": method.wrapped_mlp.n_slots,
            "tag": tag,
        })

    if 0 in eval_at:
        eval_and_log(0, tag="pre-edit")

    print(f"[grace] processing {len(edits)} edits...")
    t_stream_start = time.time()
    next_anchor_idx = 0
    while next_anchor_idx < len(eval_at) and eval_at[next_anchor_idx] <= 0:
        next_anchor_idx += 1

    for i, edit in enumerate(edits[:args.n_edits], start=1):
        method.insert(edit)
        if next_anchor_idx < len(eval_at) and i == eval_at[next_anchor_idx]:
            eval_and_log(i, tag="post-insertion")
            next_anchor_idx += 1

    t_stream = time.time() - t_stream_start
    print(f"[grace] insertion stream complete ({t_stream:.1f}s)")

    out = {
        "benchmark": args.benchmark,
        "model": args.model,
        "method": "grace",
        "seed": args.seed,
        "n_edits": args.n_edits,
        "n_holdout": len(holdout),
        "grace_layer": args.grace_layer,
        "grace_n_steps": args.grace_n_steps,
        "grace_lr": args.grace_lr,
        "grace_eps_init": args.grace_eps_init,
        "grace_init": args.grace_init,
        "n_slots_final": method.wrapped_mlp.n_slots,
        "wall_time_stream_s": t_stream,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[grace] results -> {out_path}")
    print(f"[grace] final codebook size: {method.wrapped_mlp.n_slots} slots "
          f"(from {args.n_edits} edits)")


if __name__ == "__main__":
    main()
