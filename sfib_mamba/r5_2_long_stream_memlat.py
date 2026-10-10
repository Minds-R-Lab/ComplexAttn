"""r5_2_long_stream_memlat.py -- Reviewer #5 point 2 (ESWA R5.2).

Reviewer #5 observes that the O(1) perturbation bound (Theorem~2 in the
manuscript) is per-pass, while the slot bank itself grows linearly with
the number of edits. The reviewer asks for an empirical measurement of
(a) very large edit streams, and (b) the practical memory + retrieval
overhead of the slot bank.

This script runs a single long sequential-insertion stream on
Mamba-790M + CounterFact to N = 10,000 edits (20x the paper's headline
N = 500), with evaluation at logarithmic checkpoints and
per-checkpoint telemetry: peak GPU memory, inference latency per
query, and the exact per-insertion populate cost.

Outputs:
    results/long_stream_mamba790m.json       -- per-N metrics + telemetry
    results/long_stream_mamba790m_memlog.csv -- per-N peak_mem_mb + latency_ms

Usage:
    cd sfib_mamba/
    python r5_2_long_stream_memlat.py \\
        --model state-spaces/mamba-790m-hf --layer 32 --n_edits 10000 \\
        --out results/long_stream_mamba790m.json \\
        --mem_log results/long_stream_mamba790m_memlog.csv

Expected wall time on H100-80GB: ~26 h (10,000 insertions at
~9.4 s / insertion including evaluation overhead).
"""
from __future__ import annotations

import argparse
import csv
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


def load_counterfact(seed, n_edits):
    """CounterFact loader -- same as run_shard_mamba.py uses."""
    from counterfact_data import cf_splits
    return cf_splits(n_edits=n_edits, seed=seed)


def cf_eval_batch(model, tokenizer, edits_seen, holdout, batch_size, max_new_tokens):
    """Thin wrapper around run_counterfact.cf_eval for CF."""
    from run_counterfact import cf_eval
    return cf_eval(
        model, tokenizer,
        edits_seen=edits_seen, holdout=holdout,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
    )


def measure_latency_ms(model, tokenizer, probe_prompts, max_new_tokens, repeats=3):
    """Mean / std inference latency per generate call, averaged over the
    probe set and `repeats` iterations."""
    model.eval()
    times = []
    with torch.inference_mode():
        for _ in range(repeats):
            t0 = time.time()
            for p in probe_prompts:
                ids = tokenizer(p, return_tensors="pt").to(DEVICE)
                model.generate(**ids, max_new_tokens=max_new_tokens,
                               do_sample=False, pad_token_id=tokenizer.pad_token_id)
            times.append((time.time() - t0) * 1000.0 / len(probe_prompts))
    import statistics as st
    return (st.mean(times), st.stdev(times) if len(times) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba-790m-hf")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=10000)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--v_steps", type=int, default=200)
    ap.add_argument("--n_probe", type=int, default=32,
                    help="Probe-set size for inference-latency measurement.")
    ap.add_argument("--batch_size_eval", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem_log", required=True)
    args = ap.parse_args()

    checkpoints = [50, 100, 250, 500, 1000, 2500, 5000, 10000]
    checkpoints = [n for n in checkpoints if n <= args.n_edits]
    print(f"[long_stream] device: {DEVICE}  model: {args.model}  n_edits: {args.n_edits}")
    print(f"[long_stream] checkpoints: {checkpoints}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    edits, holdout = load_counterfact(args.seed, args.n_edits)
    print(f"[long_stream] {len(edits)} edits, {len(holdout)} holdout probes")

    method = SHARDMambaMethod(
        layer_idx=args.layer, kind="out_proj",
        sim_threshold=args.tau,
        v_steps=args.v_steps, v_lr=1.0,
        v_weight_decay=0.0, v_norm_constraint=20.0,
        capture_position="prompt_last", fire_position="last",
    )
    method.setup(model, tokenizer, kb=None)

    probe_prompts = [e.prompt_template.format(e.subject) for e in holdout[:args.n_probe]]
    per_N = {}
    mem_rows = [("N", "peak_mem_mb", "n_slots", "lat_ms_mean", "lat_ms_std", "populate_s_mean")]
    insert_times = []
    t_total = time.time()
    next_cp = 0

    for i, edit in enumerate(edits, start=1):
        t_ins = time.time()
        method.insert(edit)
        insert_times.append(time.time() - t_ins)

        if next_cp < len(checkpoints) and i == checkpoints[next_cp]:
            torch.cuda.reset_peak_memory_stats()
            # Inference-latency probe
            lat_mean, lat_std = measure_latency_ms(
                model, tokenizer, probe_prompts, args.max_new_tokens, repeats=3)
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            import statistics as st
            pop_s = st.mean(insert_times[-min(len(insert_times), 100):])
            # Full eval
            result = cf_eval_batch(
                model, tokenizer, edits[:i], holdout,
                batch_size=args.batch_size_eval, max_new_tokens=args.max_new_tokens)
            per_N[i] = {
                "eff":  result["efficacy"]["accuracy"],
                "gen":  result["generalization"]["accuracy"],
                "spec": result["specificity"]["accuracy"],
                "peak_mem_mb": peak_mb,
                "n_slots": method.wrapper.n_slots if hasattr(method, "wrapper") else i,
                "lat_ms_mean": lat_mean, "lat_ms_std": lat_std,
                "populate_s_last100_mean": pop_s,
            }
            print(f"[N={i:<6}]  Eff={per_N[i]['eff']:.4f}  Gen={per_N[i]['gen']:.4f}  "
                  f"Spec={per_N[i]['spec']:.4f}  peak_mb={peak_mb:.1f}  "
                  f"lat={lat_mean:.1f}+-{lat_std:.2f}ms  pop_s={pop_s:.3f}")
            mem_rows.append((i, round(peak_mb, 2), per_N[i]["n_slots"],
                             round(lat_mean, 2), round(lat_std, 3),
                             round(pop_s, 4)))
            next_cp += 1
            # Persist after every checkpoint so a crash never loses prior work.
            with open(args.out, "w") as f:
                json.dump({"model": args.model, "layer": args.layer, "seed": args.seed,
                           "tau": args.tau, "n_edits": args.n_edits,
                           "per_N": per_N,
                           "wall_s_total": time.time() - t_total}, f, indent=2)
            with open(args.mem_log, "w", newline="") as fh:
                csv.writer(fh).writerows(mem_rows)

    print(f"\n[long_stream] wall: {(time.time() - t_total)/3600:.2f} h")
    print(f"[long_stream] wrote {args.out} and {args.mem_log}")


if __name__ == "__main__":
    main()
