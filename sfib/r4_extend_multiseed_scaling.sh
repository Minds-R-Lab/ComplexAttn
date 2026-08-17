#!/usr/bin/env bash
# r4_extend_multiseed_scaling.sh -- Reviewer #4 point 4.
#
# The R1 revision reported 3 seeds on 0.5B/1.5B and 2 seeds on 3B/7B.
# Reviewer #4 asks for a full 3-seed set across all four scales.
# This script fills in the missing seed on 3B and 7B.
#
# Runtime estimate on H100-80GB:
#   Qwen2.5-3B / seed 3   ~ 6 hr
#   Qwen2.5-7B / seed 3   ~ 12 hr
#   total                 ~ 18 hr wall time

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p results

for MODEL in "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct"; do
    python r3_scaling_multiseed.py \
        --benchmark counterfact --model "$MODEL" \
        --seed 3 --n_edits 500 --tau 0.7
done

echo "done. new files:"
ls -la results/ | grep -E "scaling.*(3b|7b).*seed3"
