#!/usr/bin/env bash
# run_shard_mamba_smoke.sh -- fast smoke test for SHARD-for-Mamba on Mamba-130m
# (the smallest available Mamba LM, 24 MambaBlock layers). Useful to verify
# the wrapper + insertion loop run end-to-end before committing to the
# full Mamba-2.8b sweep.
#
# Expected wall time on H100: ~10-15 minutes for 500 edits.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-130m-hf"
LAYER=16     # roughly two-thirds of 24-layer stack
SEED=0
N_EDITS=500
BENCH=counterfact

V_STEPS=200
V_LR=1.0
V_WD=0.0
V_NORMCAP=20.0
TAU=0.7

python run_shard_mamba.py \
  --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
  --layer "$LAYER" --kind out_proj \
  --v_steps "$V_STEPS" --v_lr "$V_LR" --v_weight_decay "$V_WD" \
  --v_norm_constraint "$V_NORMCAP" --sim_threshold "$TAU" \
  --capture_position subject_last --fire_position last \
  --out "results/${BENCH}_shard_mamba_mamba130m_seed${SEED}.json"
