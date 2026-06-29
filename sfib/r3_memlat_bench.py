"""r3_memlat_bench.py -- Reviewer #3 point R3.7 latency and memory curves.

Measures end-to-end generate() wall time and slot-bank memory footprint
as a function of bank size N. The script:

  1. Builds SHARD on a Qwen backbone.
  2. Populates the bank with N CounterFact edits (warming up the v* loop).
  3. Benchmarks `model.generate()` latency on a held-out probe set.
  4. Records peak GPU memory occupied by the bank.
  5. Repeats for each N in {10, 100, 1000, 10000}.

The cost of retrieval-only (cosine product over the bank) is included
inside `model.generate()` since the wrapped MLP runs the cosine match
during every forward pass. We report end-to-end latency rather than
retrieval-only because that's the deployment-relevant cost.

Usage on the H100 box:
    cd sfib/
    python r3_memlat_bench.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --layer 17 --out results/memlat.json

Expected wall time on a single H100-80GB: about 30 minutes (the 10000-edit
warmup is the dominant cost).
"""
from __future__ import annotations

import argparse
import json
import statistics
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

BANK_SIZES = [10, 100, 1000, 10000]
N_PROBES = 32
N_WARMUP = 5


def load_cf(seed, n_edits):
    from counterfact_data import cf_splits
    return cf_splits(n_edits=n_edits, seed=seed)


def measure_one(model_name, layer, n, seed, all_edits, probe_prompts, max_new_tokens):
    torch.cuda.empty_cache()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

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

    # Populate the bank with the first n edits
    print(f"   populating bank with {n} slots ...")
    t0 = time.time()
    for edit in all_edits[:n]:
        method.insert(edit)
    populate_s = time.time() - t0
    print(f"   bank populated in {populate_s:.1f}s")

    peak_after_insert_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                            if DEVICE.type == "cuda" else 0.0)

    # Warmup generate
    for _ in range(N_WARMUP):
        ids = tokenizer(probe_prompts[0], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model.generate(**ids, max_new_tokens=max_new_tokens,
                                do_sample=False)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark inference latency on the probe set
    samples_ms = []
    for prompt in probe_prompts:
        ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(**ids, max_new_tokens=max_new_tokens,
                                do_sample=False)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        samples_ms.append(1000.0 * (time.time() - t0))

    mean_ms = statistics.mean(samples_ms)
    std_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    print(f"   inference: {mean_ms:.1f} +/- {std_ms:.1f} ms per generate()")

    del model, method
    torch.cuda.empty_cache()
    return {
        "N": n,
        "infer_ms_mean": mean_ms,
        "infer_ms_std": std_ms,
        "populate_s": populate_s,
        "peak_mem_mb": peak_after_insert_mb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank_sizes", default="10,100,1000,10000")
    args = ap.parse_args()

    print(f"[memlat] device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[memlat] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[memlat] model={args.model}  layer={args.layer}")

    bank_sizes = [int(x) for x in args.bank_sizes.split(",")]
    max_bank = max(bank_sizes)
    all_edits, _ = load_cf(args.seed, max_bank + N_PROBES + 32)
    # Probe prompts are taken from the tail of all_edits so the bank does
    # not overlap with the probes (cleaner latency measurement: probes
    # mostly do NOT fire the bank).
    probe_prompts = [e.prompt_template.format(e.subject)
                     for e in all_edits[max_bank:max_bank + N_PROBES]]

    results = []
    for n in bank_sizes:
        print(f"\n[memlat] N = {n}")
        row = measure_one(args.model, args.layer, n, args.seed,
                          all_edits, probe_prompts, args.max_new_tokens)
        results.append(row)
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "layer": args.layer,
                       "seed": args.seed, "rows": results}, f, indent=2)
        print(f"[memlat] N={n} infer={row['infer_ms_mean']:.1f}ms "
              f"mem={row['peak_mem_mb']:.0f}MB")

    print(f"\n[memlat] wrote {args.out}")


if __name__ == "__main__":
    main()
