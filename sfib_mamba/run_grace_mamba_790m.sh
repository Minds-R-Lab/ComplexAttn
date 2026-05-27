#!/usr/bin/env bash
# run_grace_mamba_790m.sh -- GRACE-for-Mamba baseline on Mamba-790M, CF + zsRE.
# Companion to run_shard_mamba_790m.sh. Comparing the two cells gives the
# cosine-vs-Euclidean ablation on Mamba (analog of the headline finding from
# the transformer SHARD paper).
#
# Wall time per cell on H100: ~3-5 hours sequential fallback.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-790m-hf"
LAYER=32
SEED=0
N_EDITS=500

N_STEPS=100
LR=1.0
EPS_INIT=1.0
INIT=zeros

run_cell () {
  local BENCH=$1
  local OUT="results/${BENCH}_grace_mamba_mamba790m_seed${SEED}.json"
  echo "================================================================"
  echo " GRACE-for-Mamba  /  ${BENCH}  /  ${MODEL}  /  layer ${LAYER}  /  seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists."
    return 0
  fi
  python run_grace_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --n_steps "$N_STEPS" --lr "$LR" --eps_init "$EPS_INIT" --init "$INIT" \
    --out "$OUT"
}

run_cell counterfact
run_cell zsre

echo
echo "================================================================"
echo " GRACE-for-Mamba (Mamba-790M) summary @ N=500:"
echo "================================================================"
python - <<'PY'
import json, os
for path in [
    "results/counterfact_grace_mamba_mamba790m_seed0.json",
    "results/zsre_grace_mamba_mamba790m_seed0.json",
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
