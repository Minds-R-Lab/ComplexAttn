#!/usr/bin/env bash
# run_shard_mamba_multiseed_790m.sh -- multi-seed validation of SHARD-for-Mamba
# at the Mamba-790M scale.
#
# Runs SHARD-v* (200-step Adam) on CounterFact + zsRE for seeds {0, 1, 2}.
# Seed 0 is already in results/ (from run_shard_mamba_790m.sh); the runner
# auto-skips any cell whose output JSON already exists, so this script
# only does the new work.
#
# Total compute budget at H100 sequential implementation:
#   CF   seed 1  ~3.1h
#   CF   seed 2  ~3.1h
#   zsRE seed 1  ~4.2h
#   zsRE seed 2  ~4.2h
#   ------------
#   total       ~14.6h

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-790m-hf"
LAYER=32
N_EDITS=500

V_STEPS=200
V_LR=1.0
V_WD=0.0
V_NORMCAP=20.0
TAU=0.7

run_cell () {
  local BENCH=$1
  local SEED=$2
  local OUT="results/${BENCH}_shard_mamba_mamba790m_seed${SEED}.json"
  echo "================================================================"
  echo " SHARD-v* / ${BENCH} / ${MODEL} / layer ${LAYER} / seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists."
    return 0
  fi
  python run_shard_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --value_optim vstar \
    --v_steps "$V_STEPS" --v_lr "$V_LR" --v_weight_decay "$V_WD" \
    --v_norm_constraint "$V_NORMCAP" --sim_threshold "$TAU" \
    --capture_position prompt_last --fire_position last \
    --out "$OUT"
}

for SEED in 0 1 2; do
  run_cell counterfact "$SEED"
  run_cell zsre        "$SEED"
done

echo
echo "================================================================"
echo " Multi-seed SHARD-v* summary (run summarize_multiseed.py for full table):"
echo "================================================================"
python summarize_multiseed.py 2>/dev/null || \
  echo "(summarize_multiseed.py not on path; run it manually after)"
