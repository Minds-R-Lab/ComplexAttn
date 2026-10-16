#!/usr/bin/env bash
# r1_4b_grace_eps_sweep.sh -- GRACE-for-Mamba ε-sweep for R4.2 (initial range).
#
# Produces GRACE-for-Mamba's own Pareto frontier (Gen vs Spec, varying the
# deferral radius ε_init) on the same cell as the STEER τ-sweep
# (r1_4_tau_gamma_sweep.sh): CounterFact + Mamba-790M, layer 32, seed 0,
# N=500. The two sweeps overlay on the (Gen, Spec) plane to answer R4.2.
#
# GRACE uses n_steps=100 (vs STEER's 200), so each cell runs at roughly
# half the wall time: ~1.5 h/cell on H100-80GB. Full 4-cell sweep: ~6 h.

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

# Bracket GRACE's published default (eps_init=1.0) with 0.5 (tighter) and
# 2.0, 4.0 (looser). Covers an 8× range in firing radius.
for EPS in 0.5 1.0 2.0 4.0; do
  run_eps "$EPS"
done

echo "============================================================="
echo "  GRACE ε-sweep complete."
echo "============================================================="
ls -la results/grace_eps_sweep_*.json
