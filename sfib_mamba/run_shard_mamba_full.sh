#!/usr/bin/env bash
# run_shard_mamba_full.sh -- main Mamba-2.8B sweep for the SHARD-for-Mamba paper.
# Runs both SHARD and GRACE on CounterFact + zsRE at the ROMBA-reference scale.
#
# Total estimated wall time on H100 (sequential fallback): ~25-30 hours.
# With fast Mamba kernels installed: ~5-8 hours total.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-2.8b-hf"
LAYER=39        # ROMBA's strongest W_o site for 64-layer Mamba-2.8B
SEED=0
N_EDITS=500

# SHARD hyperparameters
SHARD_V_STEPS=200
SHARD_V_LR=1.0
SHARD_V_WD=0.0
SHARD_V_NORMCAP=20.0
SHARD_TAU=0.7

# GRACE hyperparameters
GRACE_N_STEPS=100
GRACE_LR=1.0
GRACE_EPS_INIT=1.0
GRACE_INIT=zeros

run_shard_cell () {
  local BENCH=$1
  local OUT="results/${BENCH}_shard_mamba_mamba2_8b_seed${SEED}.json"
  echo "================================================================"
  echo " SHARD-for-Mamba  /  ${BENCH}  /  ${MODEL}  /  layer ${LAYER}  /  seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then echo "[skip] $OUT"; return 0; fi
  python run_shard_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --v_steps "$SHARD_V_STEPS" --v_lr "$SHARD_V_LR" --v_weight_decay "$SHARD_V_WD" \
    --v_norm_constraint "$SHARD_V_NORMCAP" --sim_threshold "$SHARD_TAU" \
    --capture_position prompt_last --fire_position last \
    --out "$OUT"
}

run_grace_cell () {
  local BENCH=$1
  local OUT="results/${BENCH}_grace_mamba_mamba2_8b_seed${SEED}.json"
  echo "================================================================"
  echo " GRACE-for-Mamba  /  ${BENCH}  /  ${MODEL}  /  layer ${LAYER}  /  seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then echo "[skip] $OUT"; return 0; fi
  python run_grace_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --n_steps "$GRACE_N_STEPS" --lr "$GRACE_LR" --eps_init "$GRACE_EPS_INIT" --init "$GRACE_INIT" \
    --out "$OUT"
}

# Run SHARD first on both benchmarks, then GRACE on both.
run_shard_cell counterfact
run_shard_cell zsre
run_grace_cell counterfact
run_grace_cell zsre

echo
echo "================================================================"
echo " Mamba-2.8B SHARD vs GRACE summary @ N=500:"
echo "================================================================"
python - <<'PY'
import json, os
files = [
    ("SHARD CF",   "results/counterfact_shard_mamba_mamba2_8b_seed0.json"),
    ("SHARD zsRE", "results/zsre_shard_mamba_mamba2_8b_seed0.json"),
    ("GRACE CF",   "results/counterfact_grace_mamba_mamba2_8b_seed0.json"),
    ("GRACE zsRE", "results/zsre_grace_mamba_mamba2_8b_seed0.json"),
]
for label, path in files:
    if not os.path.exists(path):
        print(f"{label:<12} (missing)"); continue
    with open(path) as fh: d = json.load(fh)
    r500 = next((r for r in d["results"] if r["N"] == 500), None)
    if r500 is None:
        print(f"{label:<12} (no N=500)"); continue
    print(f"{label:<12}  Eff={r500['efficacy']['accuracy']:.4f}  "
          f"Gen={r500['generalization']['accuracy']:.4f}  "
          f"Spec={r500['specificity']['accuracy']:.4f}  "
          f"slots={d['n_slots_final']}")
PY
