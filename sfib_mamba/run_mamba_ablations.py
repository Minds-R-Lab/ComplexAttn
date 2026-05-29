"""run_mamba_ablations.py -- 5-preset ablation sweep for SHARD-for-Mamba.

Each preset flips one of the three GRACE-family design choices relative to
the pure SHARD configuration, plus a "shard" baseline and an "all_grace"
endpoint. Mirrors the transformer SHARD paper's centerpiece ablation
(routing / write / value-optimizer) on the Mamba codebase.

Presets:
    shard          : cosine routing + additive write + v* optim  (full SHARD)
    ablate_routing : Euclidean (GRACE-style expanding-eps) + additive + v*
    ablate_write   : cosine + substitutive + v*
    ablate_optim   : cosine + additive + vanilla Adam finetune  (no v* tricks)
    all_grace      : Euclidean + substitutive + vanilla finetune  (= GRACE-for-Mamba)

Usage:
    python run_mamba_ablations.py --benchmark counterfact \\
        --model state-spaces/mamba-790m-hf --layer 32 --seed 0 --n_edits 500 \\
        --preset shard \\
        --out results/counterfact_mamba_ablation_shard_mamba790m_seed0.json
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

import shard_mamba                          # noqa: E402, F401
import shard_mamba_realdata_patches         # noqa: E402, F401
import grace_mamba                          # noqa: E402, F401
import grace_mamba_realdata_patches         # noqa: E402, F401
from shard_mamba import SHARDMambaMethod    # noqa: E402
from grace_mamba import GRACEMambaMethod    # noqa: E402

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


def build_method(preset: str, layer: int, kind: str) -> object:
    """Construct the right Method instance for each preset."""
    if preset == "shard":
        # Pure SHARD: cosine routing + additive write + v* optim
        return SHARDMambaMethod(
            layer_idx=layer, kind=kind,
            n_v_steps=200, v_lr=1.0,
            v_weight_decay=0.0, v_norm_constraint=20.0,
            sim_threshold=0.7,
            routing="cosine", write="additive",
            capture_position="prompt_last", fire_position="last",
            value_optim="vstar",
        )
    if preset == "ablate_routing":
        # Flip routing: Euclidean expanding-eps (GRACE-style) instead of cosine.
        # Keep additive write + v* optim.
        return SHARDMambaMethod(
            layer_idx=layer, kind=kind,
            n_v_steps=200, v_lr=1.0,
            v_weight_decay=0.0, v_norm_constraint=20.0,
            routing="euclidean", expanding_eps=True, eps_init=1.0,
            write="additive",
            capture_position="prompt_last", fire_position="last",
            value_optim="vstar",
        )
    if preset == "ablate_write":
        # Flip write: substitutive instead of additive.
        # Keep cosine routing + v* optim.
        return SHARDMambaMethod(
            layer_idx=layer, kind=kind,
            n_v_steps=200, v_lr=1.0,
            v_weight_decay=0.0, v_norm_constraint=20.0,
            sim_threshold=0.7,
            routing="cosine", write="substitutive",
            capture_position="prompt_last", fire_position="last",
            value_optim="vstar",
        )
    if preset == "ablate_optim":
        # Flip optim: vanilla Adam finetune (no norm cap, no weight decay).
        # Keep cosine routing + additive write.
        return SHARDMambaMethod(
            layer_idx=layer, kind=kind,
            n_v_steps=100, v_lr=1.0,
            v_weight_decay=0.0, v_norm_constraint=1.0e10,  # effectively unconstrained
            sim_threshold=0.7,
            routing="cosine", write="additive",
            capture_position="prompt_last", fire_position="last",
            value_optim="vstar",
        )
    if preset == "all_grace":
        # All three substitutions = GRACE-for-Mamba.
        return GRACEMambaMethod(
            layer_idx=layer, kind=kind,
            n_steps=100, lr=1.0, eps_init=1.0, init="zeros",
        )
    raise ValueError(f"Unknown preset: {preset!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["counterfact", "zsre"], required=True)
    ap.add_argument("--model", default="state-spaces/mamba-790m-hf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--eval_at", default="0,1,10,50,100,250,500")
    ap.add_argument("--batch_size_eval", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--kind", choices=["out_proj", "in_proj", "x_proj"], default="out_proj")
    ap.add_argument("--preset", required=True,
                    choices=["shard", "ablate_routing", "ablate_write",
                             "ablate_optim", "all_grace"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[ablation] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[ablation] GPU: {torch.cuda.get_device_name()}")
    print(f"[ablation] preset={args.preset}  benchmark={args.benchmark}  model={args.model}")

    edits, holdout, eval_fn = load_benchmark_and_eval(
        args.benchmark, args.seed, args.n_edits)
    print(f"[ablation] loaded {len(edits)} edits, {len(holdout)} held-out specificity")
    eval_at = [int(x) for x in args.eval_at.split(",")]

    print(f"[ablation] loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    method = build_method(args.preset, layer=args.layer, kind=args.kind)
    method.setup(model, tokenizer, kb=None)

    results: list = []

    def eval_and_log(N: int, tag: str = "") -> None:
        t0 = time.time()
        result = eval_fn(model, tokenizer,
                         edits_seen=edits[:N], holdout=holdout,
                         batch_size=args.batch_size_eval,
                         max_new_tokens=args.max_new_tokens)
        eval_t = time.time() - t0
        eff  = result["efficacy"]
        gen  = result["generalization"]
        spec = result["specificity"]
        def fmt(x): return "n/a   " if x is None else f"{x:.4f}"
        print(f"  N={N:<5} "
              f"Eff={fmt(eff['accuracy'])} (n={eff['n']})   "
              f"Gen={fmt(gen['accuracy'])} (n={gen['n']})   "
              f"Spec={fmt(spec['accuracy'])} (n={spec['n']})   "
              f"[eval {eval_t:.1f}s]")
        # Use the canonical wrapper if present, else GRACE's own slot count
        n_slots = getattr(getattr(method, "wrapper", method), "n_slots", -1)
        results.append({
            "N": N, "efficacy": eff, "generalization": gen, "specificity": spec,
            "n_slots": n_slots, "tag": tag,
        })

    if 0 in eval_at:
        eval_and_log(0, tag="pre-edit")

    print(f"[ablation] processing {len(edits)} edits...")
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
    print(f"[ablation] insertion stream complete ({t_stream:.1f}s)")

    final_slots = getattr(getattr(method, "wrapper", method), "n_slots", -1)
    out = {
        "benchmark": args.benchmark, "model": args.model,
        "method": f"mamba_ablation_{args.preset}",
        "preset": args.preset,
        "seed": args.seed, "n_edits": args.n_edits,
        "n_holdout": len(holdout), "layer": args.layer, "kind": args.kind,
        "n_slots_final": final_slots,
        "wall_time_stream_s": t_stream,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ablation] results -> {out_path}")


if __name__ == "__main__":
    main()
