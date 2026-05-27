"""run_grace_mamba.py -- standalone runner for GRACEMambaMethod on CounterFact / zsRE.

Mirrors run_shard_mamba.py exactly; difference is the method used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SFIB_MAMBA_DIR = Path(__file__).parent
REPO_ROOT = SFIB_MAMBA_DIR.parent
SFIB_DIR = REPO_ROOT / "sfib"
sys.path.insert(0, str(SFIB_DIR))
sys.path.insert(0, str(SFIB_MAMBA_DIR))

import grace_mamba                                # noqa: E402, F401
import grace_mamba_realdata_patches               # noqa: E402, F401
from grace_mamba import GRACEMambaMethod          # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = SFIB_MAMBA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_benchmark_and_eval(benchmark, seed, n_edits):
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
    ap.add_argument("--model", default="state-spaces/mamba-790m-hf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--eval_at", default="0,1,10,50,100,250,500")
    ap.add_argument("--batch_size_eval", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=12)

    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--kind", choices=["out_proj", "in_proj", "x_proj"],
                    default="out_proj")
    ap.add_argument("--n_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--eps_init", type=float, default=1.0)
    ap.add_argument("--init", choices=["zeros", "random"], default="zeros")

    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[grace-mamba] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[grace-mamba] GPU: {torch.cuda.get_device_name()}")
    print(f"[grace-mamba] benchmark={args.benchmark} model={args.model}")
    print(f"[grace-mamba] {args.n_edits} edits, eval_at={args.eval_at}")

    edits, holdout, eval_fn = load_benchmark_and_eval(
        args.benchmark, args.seed, args.n_edits)
    print(f"[grace-mamba] loaded {len(edits)} edits, {len(holdout)} held-out specificity")
    eval_at = [int(x) for x in args.eval_at.split(",")]

    print(f"[grace-mamba] loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    method = GRACEMambaMethod(
        layer_idx=args.layer, kind=args.kind,
        n_steps=args.n_steps, lr=args.lr,
        eps_init=args.eps_init, init=args.init,
    )
    method.setup(model, tokenizer, kb=None)

    results: list = []

    def eval_and_log(N: int, tag: str = "") -> None:
        t0 = time.time()
        result = eval_fn(model, tokenizer,
                         edits_seen=edits[:N], holdout=holdout,
                         batch_size=args.batch_size_eval,
                         max_new_tokens=args.max_new_tokens)
        eval_t = time.time() - t0
        eff = result["efficacy"]; gen = result["generalization"]; spec = result["specificity"]
        def fmt(x): return "n/a   " if x is None else f"{x:.4f}"
        print(f"  N={N:<5} "
              f"Eff={fmt(eff['accuracy'])} (n={eff['n']})   "
              f"Gen={fmt(gen['accuracy'])} (n={gen['n']})   "
              f"Spec={fmt(spec['accuracy'])} (n={spec['n']})   "
              f"[eval {eval_t:.1f}s]")
        results.append({
            "N": N, "efficacy": eff, "generalization": gen, "specificity": spec,
            "n_slots": method.wrapper.n_slots, "tag": tag,
        })
        w = method.wrapper
        if w._diag_forwards > 0:
            dmins = w._diag_min_dist_recent
            if dmins:
                avg = sum(dmins) / len(dmins)
                print(f"    [diag] forwards={w._diag_forwards}  hits={w._diag_hits}  "
                      f"min_dist avg = {avg:.3f}  (eps_init={w.eps_init})")
            w._diag_forwards = 0
            w._diag_hits = 0
            w._diag_min_dist_recent = []

    if 0 in eval_at:
        eval_and_log(0, tag="pre-edit")

    print(f"[grace-mamba] processing {len(edits)} edits...")
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
    print(f"[grace-mamba] insertion stream complete ({t_stream:.1f}s)")

    out = {
        "benchmark": args.benchmark, "model": args.model,
        "method": "grace_mamba", "seed": args.seed, "n_edits": args.n_edits,
        "n_holdout": len(holdout), "layer": args.layer, "kind": args.kind,
        "n_steps": args.n_steps, "lr": args.lr,
        "eps_init": args.eps_init, "init": args.init,
        "n_slots_final": method.wrapper.n_slots,
        "wall_time_stream_s": t_stream,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[grace-mamba] results -> {out_path}")
    print(f"[grace-mamba] final codebook: {method.wrapper.n_slots} slots from {args.n_edits} edits")


if __name__ == "__main__":
    main()
