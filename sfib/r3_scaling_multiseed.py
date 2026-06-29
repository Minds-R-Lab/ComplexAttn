"""r3_scaling_multiseed.py -- Reviewer #3 point R3.4 multi-seed scaling.

Adds seeds 1 and 2 to the existing seed-0 Qwen2.5 scaling sweep so the
revised Section sec:scaling can report mean +/- std across 3 seeds for
the 0.5B / 1.5B / 3B / 7B sweep.

Each invocation runs ONE (model, seed) pair with the default SHARD
hyperparameters (cosine routing, additive write, v* optim, sim_threshold=0.7,
v_steps=200, v_lr=1.0). Output JSON is per-N metrics so the existing
multi-seed aggregator (in the addressable-memory branch) can pick them up
automatically.

Usage on the H100 box (single tuple):
    cd sfib/
    python r3_scaling_multiseed.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct --layer 20 --seed 1 \\
        --out results/scaling_qwen1_5b_seed1.json

Or as a driver that walks every missing (model, seed) combination:
    python r3_scaling_multiseed.py --driver

Expected wall time on a single H100-80GB per (model, seed):
    0.5B    ~  0.7 hr
    1.5B    ~  1.5 hr
    3B      ~  3   hr
    7B      ~ 11   hr
    Total seeds {1,2} across all four scales: ~32 hr.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ablations  # noqa: F401
import ablations_realdata_patches  # noqa: F401
from ablations import AblatedSHARDMethod, ABLATION_PRESETS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SFIB_DIR = Path(__file__).parent
RESULTS_DIR = SFIB_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# (model_id, layer) -- layer chosen by the existing diagnostic per scale.
SCALES = [
    ("Qwen/Qwen2.5-0.5B-Instruct", 17),
    ("Qwen/Qwen2.5-1.5B-Instruct", 20),
    ("Qwen/Qwen2.5-3B-Instruct",   26),
    ("Qwen/Qwen2.5-7B-Instruct",   20),  # 28 layers; matches the seed-0 run
]
SEEDS = [1, 2]
CHECKPOINTS = [0, 1, 10, 50, 100, 250, 500]


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


def run_one(model_name, layer, seed, benchmark, n_edits,
            batch_size_eval, max_new_tokens, out_path):
    print(f"\n== {model_name} layer={layer} seed={seed} benchmark={benchmark} ==")
    if DEVICE.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

    edits, holdout, eval_fn = load_benchmark_and_eval(benchmark, seed, n_edits)
    print(f"   {len(edits)} edits, {len(holdout)} specificity probes")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    preset = ABLATION_PRESETS["shard"]
    method = AblatedSHARDMethod(
        layer_idx=layer, sim_threshold=0.7,
        v_steps=200, v_lr=1.0,
        v_weight_decay=0.0, v_norm_constraint=20.0, eps_init=1.0,
        **preset,
    )
    method.setup(model, tokenizer, kb=None)

    per_n = {}
    t_total = time.time()
    next_anchor_idx = 0
    while next_anchor_idx < len(CHECKPOINTS) and CHECKPOINTS[next_anchor_idx] <= 0:
        # Skip 0 anchor -- a fresh frozen-model eval would just measure the
        # base accuracy and that's already on disk for seed 0.
        next_anchor_idx += 1

    for i, edit in enumerate(edits, start=1):
        method.insert(edit)
        if next_anchor_idx < len(CHECKPOINTS) and i == CHECKPOINTS[next_anchor_idx]:
            result = eval_fn(
                model, tokenizer,
                edits_seen=edits[:i], holdout=holdout,
                batch_size=batch_size_eval, max_new_tokens=max_new_tokens,
            )
            per_n[i] = {
                "eff":  result["efficacy"]["accuracy"],
                "gen":  result["generalization"]["accuracy"],
                "spec": result["specificity"]["accuracy"],
            }
            print(f"   N={i:<5} eff={per_n[i]['eff']:.4f} "
                  f"gen={per_n[i]['gen']:.4f} spec={per_n[i]['spec']:.4f}")
            next_anchor_idx += 1
            with open(out_path, "w") as f:
                json.dump({
                    "model": model_name, "layer": layer, "seed": seed,
                    "benchmark": benchmark, "n_edits": n_edits,
                    "tau": 0.7, "per_N": per_n,
                }, f, indent=2)
    wall = time.time() - t_total
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name, "layer": layer, "seed": seed,
            "benchmark": benchmark, "n_edits": n_edits,
            "tau": 0.7, "wall_s": wall, "per_N": per_n,
        }, f, indent=2)
    print(f"   wrote {out_path} ({wall:.0f}s)")


def driver(benchmark, n_edits, batch_size_eval, max_new_tokens):
    """Sequentially run every missing (model, seed) combination."""
    for (model, layer), seed in itertools.product(SCALES, SEEDS):
        tag = (model.split("/")[-1]
               .replace("Qwen2.5-", "qwen")
               .replace("-Instruct", "")
               .lower())
        out = RESULTS_DIR / f"scaling_{tag}_{benchmark}_seed{seed}.json"
        if out.exists():
            print(f"[driver] skip existing {out}")
            continue
        subprocess.run([
            sys.executable, __file__,
            "--model", model, "--layer", str(layer), "--seed", str(seed),
            "--benchmark", benchmark, "--n_edits", str(n_edits),
            "--batch_size_eval", str(batch_size_eval),
            "--max_new_tokens", str(max_new_tokens),
            "--out", str(out),
        ], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--benchmark", choices=["counterfact", "zsre"], default="counterfact")
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--batch_size_eval", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.driver:
        driver(args.benchmark, args.n_edits, args.batch_size_eval, args.max_new_tokens)
        return

    for k in ("model", "layer", "seed", "out"):
        if getattr(args, k) is None:
            ap.error(f"--{k} is required when --driver is not set")
    run_one(args.model, args.layer, args.seed,
            args.benchmark, args.n_edits,
            args.batch_size_eval, args.max_new_tokens, args.out)


if __name__ == "__main__":
    main()
