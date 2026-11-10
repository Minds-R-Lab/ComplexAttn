#!/usr/bin/env bash
# =============================================================================
# R2 revision -- Reviewer 1 point R1.2
# MQuAKE-CF-3K frozen-base baseline on Mamba-790M.
#
# Fills the `[pending H100 cell]` in manuscript Table 8 (tab:mquake) under §6.5.
# Produces one JSON cell at $RESULTS_DIR/r2_09/mquake_frozen_baseline.json
# containing single-hop and multi-hop accuracy of the UNEDITED Mamba-790M on the
# 3K-question MQuAKE-CF-3K subset we use for STEER.
#
# Why this run exists: STEER reports 1.000 single-hop and 0.127 multi-hop on
# MQuAKE-CF-3K at Mamba-790M. The reviewer asked for the paired frozen-base
# baseline so the delta is defensible (the frozen base should be near zero on
# MQuAKE's target questions; this run confirms it rather than asserts it).
#
# Wall-clock: ~45-60 minutes on one H100 (3K questions, 790M params,
# batched inference, no gradient / no edit).
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/ComplexAttn}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results_r2}"
OUT_DIR="$RESULTS_DIR/r2_09"
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

echo "[r2_09] $(date -Iseconds) start: MQuAKE frozen-base on Mamba-790M"
echo "[r2_09] REPO_ROOT=$REPO_ROOT"
echo "[r2_09] OUT_DIR=$OUT_DIR"
echo "[r2_09] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -u -m shard_mamba.eval.mquake_baseline \
    --model_name "state-spaces/mamba-790m-hf" \
    --mquake_data "data/mquake/MQuAKE-CF-3K.json" \
    --seed 0 \
    --max_new_tokens 24 \
    --batch_size 8 \
    --dtype "float16" \
    --out_json "$OUT_DIR/mquake_frozen_baseline.json"

echo "[r2_09] $(date -Iseconds) done"
echo
echo "Result:"
python -c "
import json, sys
with open('$OUT_DIR/mquake_frozen_baseline.json') as f:
    d = json.load(f)
print(f\"  single-hop (Eff): {d['single_hop_acc']:.4f}\")
print(f\"  multi-hop  (Acc): {d['multi_hop_acc']:.4f}\")
print(f\"  N questions: {d['n_questions']}\")
"
echo
echo "Next: patch manuscript.tex tab:mquake, replacing [pending H100 cell]"
echo "with the single_hop_acc and multi_hop_acc values printed above."
