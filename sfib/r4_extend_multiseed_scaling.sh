#!/usr/bin/env bash
# r4_extend_multiseed_scaling.sh -- Reviewer #4 point 4.
#
# R1 shipped 3 seeds on 0.5B/1.5B and 2 seeds ({1, 2}) on 3B/7B.
# Reviewer #4 asks for a full 3-seed set across all four scales.
# We add seed 0 (matching the seed-0 present on 0.5B and 1.5B) rather
# than a new seed 3, so the resulting 3-seed set on 3B and 7B uses the
# same seed indices {0, 1, 2} as the other two scales.
#
# Runtime estimate on H100-80GB (from r3_scaling_multiseed.py docstring):
#   Qwen2.5-3B / seed 0   ~  3 hr
#   Qwen2.5-7B / seed 0   ~ 11 hr
#   total                 ~ 14 hr wall time
#
# The r3 driver skips (model, seed) combinations whose output JSON
# already exists, so re-running this script after an interruption is
# safe.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p results

# (model, layer, out-tag) tuples; tags mirror the existing seed 1/2 files
declare -a JOBS=(
    "Qwen/Qwen2.5-3B-Instruct|26|scaling_qwen3b_counterfact_seed0.json"
    "Qwen/Qwen2.5-7B-Instruct|20|scaling_qwen7b_counterfact_seed0.json"
)

for J in "${JOBS[@]}"; do
    IFS='|' read -r MODEL LAYER OUT <<< "$J"
    if [[ -f "results/$OUT" ]]; then
        echo "[skip] results/$OUT already exists"
        continue
    fi
    echo "==============================================="
    echo "  MODEL=$MODEL  LAYER=$LAYER  seed=0"
    echo "  writing results/$OUT"
    echo "==============================================="
    python r3_scaling_multiseed.py \
        --model "$MODEL" --layer "$LAYER" --seed 0 \
        --benchmark counterfact --n_edits 500 \
        --out "results/$OUT"
done

echo
echo "done. new files:"
ls -la results/ | grep -E "scaling_qwen(3b|7b).*seed0"
