"""r3_long_stream.py -- Reviewer #3 point R3.7 longer streams.

Extends the SHARD insertion stream to 1k, 5k, 10k edits on Qwen2.5-0.5B +
CounterFact and records:
  - Eff / Gen / Spec at a log-spaced set of checkpoints
  - per-insertion wall time (CSV)
  - per-insertion peak GPU memory (CSV)

The output JSON + CSV combination feeds the new Section sec:scalability of
the revised manuscript. CounterFact has ~21k edit records, so 10k fits in
the dataset without resampling.

Usage on the H100 box:
    cd sfib/
    python r3_long_stream.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct --layer 17 \\
        --n_edits 10000 \\
        --out results/long_stream_qwen0_5b.json \\
        --mem_log results/long_stream_qwen0_5b_memlog.csv

Expected wall time on a single H100-80GB: ~5 hr for the full 10k.
"""
from __future__ import annotations

import argparse
import json
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

# Log-spaced eval anchors
CHECKPOINTS = [50, 100, 250, 500, 1000, 2500, 5000, 10000]


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
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--benchmark", choices=["counterfact", "zsre"], default="counterfact")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=10000)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--batch_size_eval", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem_log", default=None,
                    help="optional CSV path for per-insertion wall time + memory")
    args = ap.parse_args()

    print(f"[long_stream] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[long_stream] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[long_stream] benchmark={args.benchmark}  model={args.model}  "
          f"layer={args.layer}  n_edits={args.n_edits}")

    edits, holdout, eval_fn = load_benchmark_and_eval(
        args.benchmark, args.seed, args.n_edits,
    )
    print(f"[long_stream] {len(edits)} edits, {len(holdout)} specificity probes")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    preset = ABLATION_PRESETS["shard"]
    method = AblatedSHARDMethod(
        layer_idx=args.layer, sim_threshold=args.tau,
        v_steps=200, v_lr=1.0,
        v_weight_decay=0.0, v_norm_constraint=20.0, eps_init=1.0,
        **preset,
    )
    method.setup(model, tokenizer, kb=None)

    mem_log = open(args.mem_log, "w") if args.mem_log else None
    if mem_log:
        mem_log.write("step,insert_s,peak_mem_mb\n")

    per_n = {}
    t_total = time.time()
    next_anchor_idx = 0
    while next_anchor_idx < len(CHECKPOINTS) and CHECKPOINTS[next_anchor_idx] > args.n_edits:
        next_anchor_idx += 1
    # Reset to start of checkpoint list, but we'll skip anchors beyond n_edits at eval time.

    next_anchor_idx = 0

    for i, edit in enumerate(edits[:args.n_edits], start=1):
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_ins = time.time()
        method.insert(edit)
        ins_s = time.time() - t_ins
        peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                   if DEVICE.type == "cuda" else 0.0)

        if mem_log:
            mem_log.write(f"{i},{ins_s:.4f},{peak_mb:.1f}\n")
            if i % 100 == 0:
                mem_log.flush()

        while (next_anchor_idx < len(CHECKPOINTS)
               and CHECKPOINTS[next_anchor_idx] == i):
            print(f"[long_stream] eval at N={i} ({(time.time()-t_total)/60:.1f} min in)")
            result = eval_fn(
                model, tokenizer,
                edits_seen=edits[:i], holdout=holdout,
                batch_size=args.batch_size_eval, max_new_tokens=args.max_new_tokens,
            )
            per_n[i] = {
                "eff":  result["efficacy"]["accuracy"],
                "gen":  result["generalization"]["accuracy"],
                "spec": result["specificity"]["accuracy"],
                "peak_mem_mb": peak_mb,
            }
            print(f"   eff={per_n[i]['eff']:.4f} gen={per_n[i]['gen']:.4f} "
                  f"spec={per_n[i]['spec']:.4f}   peak_mem={peak_mb:.0f}MB")
            next_anchor_idx += 1
            with open(args.out, "w") as f:
                json.dump({
                    "model": args.model, "layer": args.layer, "seed": args.seed,
                    "tau": args.tau, "benchmark": args.benchmark,
                    "n_edits": args.n_edits, "per_N": per_n,
                }, f, indent=2)

    if mem_log:
        mem_log.close()
    print(f"[long_stream] wrote {args.out}")


if __name__ == "__main__":
    main()
