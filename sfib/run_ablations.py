"""run_ablations.py -- runs an AblatedSHARDMethod configuration on CounterFact or zsRE.

Five canonical experiment names (preset configurations defined in ablations.py):
  shard, ablate_routing, ablate_write, ablate_optim, all_grace

Each preset toggles one or more of SHARD's three design substitutions. Comparing
the four single-flip experiments against `shard` tells us which substitution
is responsible for the Pareto-frontier shift away from GRACE.

Usage:
  python run_ablations.py --benchmark counterfact --model Qwen/Qwen2.5-0.5B-Instruct \
      --layer 17 --preset ablate_routing --seed 0 --n_edits 500 \
      --out results/counterfact_ablate_routing_qwen0_5b_seed0.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ablations                              # noqa: F401  (registers)
import ablations_realdata_patches             # noqa: F401  (CF/zsRE patches)
from ablations import AblatedSHARDMethod, ABLATION_PRESETS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SFIB_DIR = Path(__file__).parent
RESULTS_DIR = SFIB_DIR / "results"
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
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--eval_at", default="0,1,10,50,100,250,500")
    ap.add_argument("--batch_size_eval", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=12)

    ap.add_argument("--preset", choices=list(ABLATION_PRESETS.keys()), required=True,
                    help="One of: shard, ablate_routing, ablate_write, ablate_optim, all_grace")
    ap.add_argument("--layer", type=int, default=17)
    # Hyperparameters (match SHARD/SwiGLU defaults; override per cell)
    ap.add_argument("--v_steps", type=int, default=200)
    ap.add_argument("--v_lr", type=float, default=1.0)
    ap.add_argument("--v_weight_decay", type=float, default=0.0)
    ap.add_argument("--v_norm_constraint", type=float, default=20.0)
    ap.add_argument("--sim_threshold", type=float, default=0.7)
    ap.add_argument("--eps_init", type=float, default=1.0)

    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preset = ABLATION_PRESETS[args.preset]
    print(f"[abl] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[abl] GPU: {torch.cuda.get_device_name()}")
    print(f"[abl] preset={args.preset}  -> routing={preset['routing']}, "
          f"write_mode={preset['write_mode']}, value_optim={preset['value_optim']}")
    print(f"[abl] benchmark={args.benchmark}  model={args.model}  layer={args.layer}")

    edits, holdout, eval_fn = load_benchmark_and_eval(
        args.benchmark, args.seed, args.n_edits)
    print(f"[abl] loaded {len(edits)} edits, {len(holdout)} held-out specificity")

    eval_at = [int(x) for x in args.eval_at.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    method = AblatedSHARDMethod(
        layer_idx=args.layer,
        routing=preset["routing"],
        write_mode=preset["write_mode"],
        value_optim=preset["value_optim"],
        v_steps=args.v_steps,
        v_lr=args.v_lr,
        v_weight_decay=args.v_weight_decay,
        v_norm_constraint=args.v_norm_constraint,
        sim_threshold=args.sim_threshold,
        eps_init=args.eps_init,
    )
    method.setup(model, tokenizer, kb=None)

    results = []

    def eval_and_log(N, tag=""):
        t0 = time.time()
        result = eval_fn(model, tokenizer,
                          edits_seen=edits[:N], holdout=holdout,
                          batch_size=args.batch_size_eval,
                          max_new_tokens=args.max_new_tokens)
        eval_t = time.time() - t0
        eff = result["efficacy"]; gen = result["generalization"]; spec = result["specificity"]
        def fmt(x): return "n/a   " if x is None else f"{x:.4f}"
        print(f"  N={N:<5} Eff={fmt(eff['accuracy'])} (n={eff['n']})   "
              f"Gen={fmt(gen['accuracy'])} (n={gen['n']})   "
              f"Spec={fmt(spec['accuracy'])} (n={spec['n']})   [eval {eval_t:.1f}s]")
        results.append({"N": N, "efficacy": eff, "generalization": gen,
                         "specificity": spec, "n_slots": method.wrapped_mlp.n_slots,
                         "tag": tag})

    if 0 in eval_at:
        eval_and_log(0, tag="pre-edit")

    t0 = time.time()
    next_a = 0
    while next_a < len(eval_at) and eval_at[next_a] <= 0:
        next_a += 1
    for i, edit in enumerate(edits[:args.n_edits], start=1):
        method.insert(edit)
        if next_a < len(eval_at) and i == eval_at[next_a]:
            eval_and_log(i, tag="post-insertion")
            next_a += 1
    t_stream = time.time() - t0
    print(f"[abl] insertion stream complete ({t_stream:.1f}s)")

    out = {
        "benchmark": args.benchmark,
        "model": args.model,
        "method": "ablated_shard",
        "preset": args.preset,
        "routing": preset["routing"],
        "write_mode": preset["write_mode"],
        "value_optim": preset["value_optim"],
        "seed": args.seed,
        "n_edits": args.n_edits,
        "n_holdout": len(holdout),
        "layer": args.layer,
        "v_steps": args.v_steps,
        "v_lr": args.v_lr,
        "v_weight_decay": args.v_weight_decay,
        "v_norm_constraint": args.v_norm_constraint,
        "sim_threshold": args.sim_threshold,
        "eps_init": args.eps_init,
        "n_slots_final": method.wrapped_mlp.n_slots,
        "wall_time_stream_s": t_stream,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[abl] results -> {out_path}")


if __name__ == "__main__":
    main()
