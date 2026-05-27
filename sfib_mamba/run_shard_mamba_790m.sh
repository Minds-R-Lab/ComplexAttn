#!/usr/bin/env bash
# run_shard_mamba_790m.sh -- intermediate-scale verification of SHARD-for-Mamba
# on Mamba-790M (48 MambaBlock layers, d_inner=3072, d_model=1536), which is
# the sweet spot between the 130M smoke test (~1.5h) and the 2.8B reference
# (~30h). At 790M the wall time should be 6-9 hours per cell.
#
# We run on both CounterFact and zsRE, layer 32 (~67% depth), with the same
# hyperparameters that worked on the 130M smoke test (v_steps=200, v_lr=1.0,
# tau=0.7).

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-790m-hf"
LAYER=32
SEED=0
N_EDITS=500

V_STEPS=200
V_LR=1.0
V_WD=0.0
V_NORMCAP=20.0
TAU=0.7

run_cell () {
  local BENCH=$1
  local OUT="results/${BENCH}_shard_mamba_mamba790m_seed${SEED}.json"
  echo "================================================================"
  echo " SHARD-for-Mamba  /  ${BENCH}  /  ${MODEL}  /  layer ${LAYER}  /  seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists."
    return 0
  fi
  python run_shard_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --v_steps "$V_STEPS" --v_lr "$V_LR" --v_weight_decay "$V_WD" \
    --v_norm_constraint "$V_NORMCAP" --sim_threshold "$TAU" \
    --capture_position prompt_last --fire_position last \
    --out "$OUT"
}

run_cell counterfact
run_cell zsre

echo
echo "================================================================"
echo " Mamba-790M SHARD summary @ N=500:"
echo "================================================================"
python - <<'PY'
import json, os
for path in [
    "results/counterfact_shard_mamba_mamba790m_seed0.json",
    "results/zsre_shard_mamba_mamba790m_seed0.json",
]:
    if not os.path.exists(path):
        print(f"{path}: (missing)"); continue
    with open(path) as fh: d = json.load(fh)
    r500 = next((r for r in d["results"] if r["N"] == 500), None)
    if r500 is None:
        print(f"{path}: (no N=500)"); continue
    print(f"{path}: "
          f"Eff={r500['efficacy']['accuracy']:.4f}  "
          f"Gen={r500['generalization']['accuracy']:.4f}  "
          f"Spec={r500['specificity']['accuracy']:.4f}  "
          f"slots={d['n_slots_final']}")
PY
