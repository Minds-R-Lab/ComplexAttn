#!/usr/bin/env bash
# run_v2_all.sh -- orchestrator for the NEUCOM-D-26-11036 revision
# experiments. Runs the cheap diagnostics first so they can be folded into
# response_to_reviewers.tex quickly, then the expensive long runs.
#
# Outputs land in results/. All scripts re-resume gracefully if you stop
# them mid-run and re-launch -- existing JSONs are skipped at the cell
# level. The .json files are the same shape as the seed-0 v1 results so
# the aggregator picks them up without modification.

set -e
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p results

# ----------------------------------------------------------------------
# Tier 0 -- cheap diagnostics (under 1 hr each on H100)
# ----------------------------------------------------------------------

if [[ ! -f results/distance_distributions_qwen0_5b.json ]]; then
  python r3_distance_distributions.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --layer 17 --n_edits 500 \
    --out results/distance_distributions_qwen0_5b.json
fi

if [[ ! -f results/fp_fn_qwen0_5b.json ]]; then
  python r3_fp_fn_analysis.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --layer 17 --n_edits 500 \
    --out results/fp_fn_qwen0_5b.json
fi

if [[ ! -f results/memlat_qwen0_5b.json ]]; then
  python r3_memlat_bench.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --layer 17 \
    --out results/memlat_qwen0_5b.json
fi

# ----------------------------------------------------------------------
# Tier 1 -- moderate (4-6 hr each)
# ----------------------------------------------------------------------

if [[ ! -f results/tau_pareto_qwen0_5b_cf.json ]]; then
  python r3_tau_sweep_pareto.py --benchmark counterfact \
    --model Qwen/Qwen2.5-0.5B-Instruct --layer 17 \
    --out results/tau_pareto_qwen0_5b_cf.json
fi

if [[ ! -f results/tau_pareto_qwen1_5b_cf.json ]]; then
  python r3_tau_sweep_pareto.py --benchmark counterfact \
    --model Qwen/Qwen2.5-1.5B-Instruct --layer 20 \
    --out results/tau_pareto_qwen1_5b_cf.json
fi

if [[ ! -f results/composition_oracle.json ]]; then
  python r2_composition_oracle.py \
    --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
             Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct \
    --out results/composition_oracle.json
fi

# ----------------------------------------------------------------------
# Tier 2 -- expensive (5-32 hr)
# ----------------------------------------------------------------------

if [[ ! -f results/long_stream_qwen0_5b.json ]]; then
  python r3_long_stream.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --layer 17 --n_edits 10000 \
    --out results/long_stream_qwen0_5b.json \
    --mem_log results/long_stream_qwen0_5b_memlog.csv
fi

# Multi-seed scaling: driver mode walks every (model, seed) combination.
python r3_scaling_multiseed.py --driver --benchmark counterfact

if [[ ! -f results/composition_shard_7b.json ]]; then
  python r2_composition_shard_7b.py \
    --model Qwen/Qwen2.5-7B-Instruct --layer 20 \
    --n_edits 500 \
    --out results/composition_shard_7b.json
fi

echo
echo "================================================================"
echo " V2 revision experiments complete. Result files:"
echo "================================================================"
ls -la results/ | grep -E "tau_pareto|distance|fp_fn|memlat|long_stream|scaling.*seed[12]|composition_(oracle|shard_7b)"
