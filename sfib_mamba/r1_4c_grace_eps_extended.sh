#!/usr/bin/env bash
# r1_4c_grace_eps_extended.sh -- extend GRACE ε-sweep to the loose-firing regime.
#
# Rules out "GRACE could reach STEER's Spec regime with larger ε" by direct
# test. Companion to r1_4b (which runs ε=0.5-4). ~1.5 h/cell × 3 cells = ~5 h.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results

BENCH=counterfact
MODEL=state-spaces/mamba-790m-hf
LAYER=32
SEED=0
N_EDITS=500

run_eps () {
  local EPS=$1
  local OUT="results/grace_eps_sweep_${EPS}_${BENCH}_mamba790m_seed${SEED}.json"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists"
    return
  fi
  echo "============================================================="
  echo " GRACE  eps_init=${EPS}  layer=${LAYER}  seed=${SEED}"
  echo "============================================================="
  python run_grace_mamba.py \
    --benchmark "$BENCH" \
    --model    "$MODEL" \
    --seed     "$SEED" \
    --n_edits  "$N_EDITS" \
    --layer    "$LAYER" \
    --kind     out_proj \
    --n_steps  100 \
    --lr       1.0 \
    --eps_init "$EPS" \
    --init     zeros \
    --out      "$OUT"
}

for EPS in 8.0 16.0 32.0; do
  run_eps "$EPS"
done

echo "============================================================="
echo "  GRACE ε-sweep extended complete."
echo "============================================================="
ls -la results/grace_eps_sweep_*.json
