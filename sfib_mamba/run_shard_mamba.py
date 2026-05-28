"""run_shard_mamba.py -- standalone runner for SHARDMambaMethod on CounterFact / zsRE.

Mirrors run_grace.py / run_ablations.py: imports cf_eval and zsre_eval from
the existing CF/zsRE runners so the metric definitions are identical to the
transformer numbers in the SHARD paper.

Usage:
    python run_shard_mamba.py --benchmark counterfact \\
        --model state-spaces/mamba-790m-hf --layer 24 --seed 0 --n_edits 500 \\
        --out results/counterfact_shard_mamba_mamba790m_seed0.json

Default layers (rough middle / two-thirds of the stack):
    Mamba-130m  (24 layers)  -> 16
    Mamba-790m  (48 layers)  -> 32
    Mamba-1.4b  (48 layers)  -> 32
    Mamba-2.8b  (64 layers)  -> 39   (ROMBA's strongest single-edit layer for W_o)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make the sfib package importable so we can reuse cf_eval / zsre_eval.
SFIB_MAMBA_DIR = Path(__file__).parent
REPO_ROOT = SFIB_MAMBA_DIR.parent
SFIB_DIR = REPO_ROOT / "sfib"
sys.path.insert(0, str(SFIB_DIR))
sys.path.insert(0, str(SFIB_MAMBA_DIR))

import shard_mamba                              # noqa: E402, F401  (registers)
import shard_mamba_realdata_patches             # noqa: E402, F401  (CF/zsRE patches)
from shard_mamba import SHARDMambaMethod        # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = SFIB_MAMBA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


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
    ap.add_argument("--model", default="state-spaces/mamba-790m-hf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--eval_at", default="0,1,10,50,100,250,500")
    ap.add_argument("--batch_size_eval", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=12)

    ap.add_argument("--layer", type=int, default=32,
                    help="Mamba layer index to wrap (default 32 -- mid-stack for 48-layer models)")
    ap.add_argument("--kind", choices=["out_proj", "in_proj", "x_proj"],
                    default="out_proj",
                    help="Which MambaBlock projection to wrap (W_o by default, ROMBA's best site)")
    ap.add_argument("--v_steps", type=int, default=200)
    ap.add_argument("--v_lr", type=float, default=1.0)
    ap.add_argument("--v_weight_decay", type=float, default=0.0)
    ap.add_argument("--v_norm_constraint", type=float, default=20.0)
    ap.add_argument("--sim_threshold", type=float, default=0.7)
    ap.add_argument("--max_slots", type=int, default=8000)
    ap.add_argument("--capture_position", choices=["subject_last", "prompt_last"],
                    default="prompt_last",
                    help="Where to capture the slot key (must match fire_position for the slot to actually fire)")
    ap.add_argument("--fire_position", choices=["last", "all"], default="last")

    ap.add_argument("--value_optim", choices=["vstar", "lqr", "lqr_gn"], default="vstar",
                    help="delta_v optimizer: vstar (ROME-style 200-step Adam), "
                         "lqr (one-shot saturated control), or "
                         "lqr_gn (multi-step Gauss-Newton LQR)")
    ap.add_argument("--lqr_lambda", type=float, default=1e-3,
                    help="LQR Tikhonov weight (scaled by 1/||v_orig||^2); only used when value_optim=lqr")
    ap.add_argument("--lqr_alpha_scale", type=float, default=1.0,
                    help="Scale on the LQR feedback gain (1.0 = saturate at the box boundary)")
    ap.add_argument("--n_lqr_iters", type=int, default=5,
                    help="Number of Gauss-Newton LQR iterations; only used when value_optim=lqr_gn (default 5)")

    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[shard-mamba] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[shard-mamba] GPU: {torch.cuda.get_device_name()}")
    print(f"[shard-mamba] benchmark={args.benchmark} model={args.model}")
    print(f"[shard-mamba] {args.n_edits} edits, eval_at={args.eval_at}")

    edits, holdout, eval_fn = load_benchmark_and_eval(
        args.benchmark, args.seed, args.n_edits)
    print(f"[shard-mamba] loaded {len(edits)} edits, {len(holdout)} held-out specificity")

    eval_at = [int(x) for x in args.eval_at.split(",")]

    print(f"[shard-mamba] loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    method = SHARDMambaMethod(
        layer_idx=args.layer, kind=args.kind,
        n_v_steps=args.v_steps, v_lr=args.v_lr,
        v_weight_decay=args.v_weight_decay,
        v_norm_constraint=args.v_norm_constraint,
        sim_threshold=args.sim_threshold,
        max_slots=args.max_slots,
        capture_position=args.capture_position,
        fire_position=args.fire_position,
        value_optim=args.value_optim,
        lqr_lambda=args.lqr_lambda,
        lqr_alpha_scale=args.lqr_alpha_scale,
        n_lqr_iters=args.n_lqr_iters,
    )
    method.setup(model, tokenizer, kb=None)

    results: list = []

    def eval_and_log(N: int, tag: str = "") -> None:
        t0 = time.time()
        result = eval_fn(
            model, tokenizer,
            edits_seen=edits[:N], holdout=holdout,
            batch_size=args.batch_size_eval,
            max_new_tokens=args.max_new_tokens,
        )
        eval_t = time.time() - t0
        eff = result["efficacy"]
        gen = result["generalization"]
        spec = result["specificity"]
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
        # Diagnostic: dump the routing stats so we can see if slots ever fire.
        w = method.wrapper
        if hasattr(w, "_diag_forwards") and w._diag_forwards > 0:
            sims = w._diag_max_sim_recent
            if sims:
                import statistics
                avg = sum(sims) / len(sims)
                mx = max(sims)
                mn = min(sims)
                print(f"    [diag] forwards={w._diag_forwards}  hits={w._diag_hits}  "
                      f"max_sim avg/min/max = {avg:.3f}/{mn:.3f}/{mx:.3f}  (tau={w.sim_threshold})")
            # Reset for the next anchor window
            w._diag_forwards = 0
            w._diag_hits = 0
            w._diag_max_sim_recent = []

    if 0 in eval_at:
        eval_and_log(0, tag="pre-edit")

    print(f"[shard-mamba] processing {len(edits)} edits...")
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
    print(f"[shard-mamba] insertion stream complete ({t_stream:.1f}s)")

    out = {
        "benchmark": args.benchmark, "model": args.model,
        "method": "shard_mamba", "seed": args.seed, "n_edits": args.n_edits,
        "n_holdout": len(holdout), "layer": args.layer, "kind": args.kind,
        "v_steps": args.v_steps, "v_lr": args.v_lr,
        "v_weight_decay": args.v_weight_decay,
        "v_norm_constraint": args.v_norm_constraint,
        "sim_threshold": args.sim_threshold,
        "capture_position": args.capture_position,
        "fire_position": args.fire_position,
        "value_optim": args.value_optim,
        "lqr_lambda": args.lqr_lambda,
        "lqr_alpha_scale": args.lqr_alpha_scale,
        "n_lqr_iters": args.n_lqr_iters,
        "n_slots_final": method.wrapper.n_slots,
        "wall_time_stream_s": t_stream,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkd