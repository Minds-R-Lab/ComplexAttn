#!/usr/bin/env bash
# run_shard_mamba_lqr_gn10_multiseed_790m.sh -- multi-seed validation of the
# LQR-GN10 variant on Mamba-790M.
#
# Same cells (CF + zsRE) and seeds (0, 1, 2) as the v* multi-seed runner,
# but using value_optim=lqr_gn with n_lqr_iters=10. Roughly 20x cheaper
# per cell than v*, so the full multi-seed sweep runs in under an hour.
#
# Per-cell wall times (extrapolated from seed-0 timings):
#   CF   seed 1  ~10 min
#   CF   seed 2  ~10 min
#   zsRE seed 1  ~14 min
#   zsRE seed 2  ~14 min
#   ------------
#   total       ~48 min  (plus the already-completed seed 0)

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-790m-hf"
LAYER=32
N_EDITS=500

V_NORMCAP=20.0
TAU=0.7

LQR_ALPHA_SCALE=1.0
N_LQR_ITERS=10

run_cell () {
  local BENCH=$1
  local SEED=$2
  local OUT="results/${BENCH}_shard_mamba_lqr_gn10_mamba790m_seed${SEED}.json"
  echo "================================================================"
  echo " SHARD-LQR-GN10 / ${BENCH} / ${MODEL} / layer ${LAYER} / iters ${N_LQR_ITERS} / seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists."
    return 0
  fi
  python run_shard_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --value_optim lqr_gn \
    --n_lqr_iters "$N_LQR_ITERS" \
    --lqr_alpha_scale "$LQR_ALPHA_SCALE" \
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
echo " Multi-seed SHARD-LQR-GN10 summary:"
echo "================================================================"
python summarize_multiseed.py 2>/dev/null || \
  echo "(summarize_multiseed.py not on path; run it manually after)"
