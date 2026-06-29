"""r3_tau_sweep_pareto.py -- Reviewer #3 point R3.3 Pareto sweep.

Sweeps SHARD's sim_threshold (tau) and GRACE's eps_init across the same
(model, benchmark) cell, runs the full 500-edit insertion stream for each
grid point, and records efficacy / generalization / specificity at N=500.

The resulting JSON backs the new Section sec:tau-pareto in the revised
NEUCOM-D-26-11036 manuscript. Reviewer #3 asked for evidence that SHARD and
GRACE occupy distinct points on a generalization-specificity frontier rather
than one dominating the other; this script produces that frontier directly.

Usage on the H100 box:
    cd sfib/
    python r3_tau_sweep_pareto.py --benchmark counterfact \\
        --model Qwen/Qwen2.5-0.5B-Instruct --layer 17 \\
        --out results/tau_pareto_qwen0_5b_cf.json

    python r3_tau_sweep_pareto.py --benchmark counterfact \\
        --model Qwen/Qwen2.5-1.5B-Instruct --layer 20 \\
        --out results/tau_pareto_qwen1_5b_cf.json

Default grids:
    SHARD tau in {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}
    GRACE eps in {0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0}

Expected wall time on a single H100-80GB: about 6 hours per cell.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ablations  # noqa: F401  (registers AblatedSHARDMethod)
import ablations_realdata_patches  # noqa: F401  (CF/zsRE rewrite-builder patches)
import grace_method  # noqa: F401
import grace_realdata_patches  # noqa: F401
from ablations import AblatedSHARDMethod, ABLATION_PRESETS
from grace_method import GRACEMethod

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SFIB_DIR = Path(__file__).parent
RESULTS_DIR = SFIB_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TAU_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
EPS_GRID = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]


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


def run_shard(model_name, edits, holdout, eval_fn, tau, layer, n_v_steps,
              v_lr, batch_size_eval, max_new_tokens):
    """Run AblatedSHARDMethod with cosine routing at the given tau."""
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    preset = ABLATION_PRESETS["shard"]
    method = AblatedSHARDMethod(
        layer_idx=layer, sim_threshold=tau,
        v_steps=n_v_steps, v_lr=v_lr,
        v_weight_decay=0.0, v_norm_constraint=20.0, eps_init=1.0,
        **preset,
    )
    method.setup(model, tokenizer, kb=None)

    for edit in edits:
        method.insert(edit)

    result = eval_fn(
        model, tokenizer,
        edits_seen=edits, holdout=holdout,
        batch_size=batch_size_eval, max_new_tokens=max_new_tokens,
    )
    out = {
        "method": "SHARD", "threshold": tau,
        "eff":  result["efficacy"]["accuracy"],
        "gen":  result["generalization"]["accuracy"],
        "spec": result["specificity"]["accuracy"],
    }
    del model, method
    torch.cuda.empty_cache()
    return out


def run_grace(model_name, edits, holdout, eval_fn, eps, layer, n_steps,
              lr, batch_size_eval, max_new_tokens):
    """Run the published GRACE baseline at the given eps_init radius."""
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    method = GRACEMethod(
        layer_idx=layer, n_steps=n_steps, lr=lr,
        eps_init=eps, init="zeros",
    )
    method.setup(model, tokenizer, kb=None)

    for edit in edits:
        method.insert(edit)

    result = eval_fn(
        model, tokenizer,
        edits_seen=edits, holdout=holdout,
        batch_size=batch_size_eval, max_new_tokens=max_new_tokens,
    )
    out = {
        "method": "GRACE", "threshold": eps,
        "eff":  result["efficacy"]["accuracy"],
        "gen":  result["generalization"]["accuracy"],
        "spec": result["specificity"]["accuracy"],
    }
    del model, method
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["counterfact", "zsre"], default="counterfact")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=17,
                    help="MLP edit layer; 17 for Qwen2.5-0.5B, 20 for 1.5B, "
                         "26 for 3B, 13 for TinyLlama")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=500)
    ap.add_argument("--batch_size_eval", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--shard_v_steps", type=int, default=200)
    ap.add_argument("--shard_v_lr", type=float, default=1.0)
    ap.add_argument("--grace_n_steps", type=int, default=100)
    ap.add_argument("--grace_lr", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="shard,grace",
                    help="comma-separated subset of {shard, grace} to run")
    args = ap.parse_args()

    print(f"[tau_sweep] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[tau_sweep] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[tau_sweep] benchmark={args.benchmark}  model={args.model}  layer={args.layer}")

    edits, holdout, eval_fn = load_benchmark_and_eval(
        args.benchmark, args.seed, args.n_edits,
    )
    print(f"[tau_sweep] {len(edits)} edits, {len(holdout)} specificity probes")

    rows = []

    methods = {m.strip() for m in args.methods.split(",")}

    if "shard" in methods:
        for tau in TAU_GRID:
            t0 = time.time()
            print(f"\n[SHARD] tau = {tau:.2f}")
            row = run_shard(
                args.model, edits, holdout, eval_fn, tau, args.layer,
                args.shard_v_steps, args.shard_v_lr,
                args.batch_size_eval, args.max_new_tokens,
            )
            row["wall_s"] = time.time() - t0
            print(f"  eff={row['eff']:.4f}  gen={row['gen']:.4f}  spec={row['spec']:.4f}  "
                  f"[{row['wall_s']:.0f}s]")
            rows.append(row)
            with open(args.out, "w") as f:
                json.dump({"benchmark": args.benchmark, "model": args.model,
                           "seed": args.seed, "rows": rows}, f, indent=2)

    if "grace" in methods:
        for eps in EPS_GRID:
            t0 = time.time()
            print(f"\n[GRACE] eps = {eps:.2f}")
            row = run_grace(
                args.model, edits, holdout, eval_fn, eps, args.layer,
                args.grace_n_steps, args.grace_lr,
                args.batch_size_eval, args.max_new_tokens,
            )
            row["wall_s"] = time.time() - t0
            print(f"  eff={row['eff']:.4f}  gen={row['gen']:.4f}  spec={row['spec']:.4f}  "
                  f"[{row['wall_s']:.0f}s]")
            rows.append(row)
            with open(args.out, "w") as f:
                json.dump({"benchmark": args.benchmark, "model": args.model,
                           "seed": args.seed, "rows": rows}, f, indent=2)

    print(f"\n[tau_sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
