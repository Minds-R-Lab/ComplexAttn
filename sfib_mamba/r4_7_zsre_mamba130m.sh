#!/usr/bin/env bash
# r4_7_zsre_mamba130m.sh -- Reviewer #4 point 7 (ESWA R4.7).
#
# Reviewer #4 observes that zsRE results are reported only for Mamba-790M
# and Mamba-2.8B, while the contribution claim covers all three Mamba
# scales (130M, 790M, 2.8B). This script fills in the missing cell:
# STEER (SHARD-v*) on zsRE at Mamba-130M, layer 16, seed 0.
#
# Mamba-130M is small, so this is cheap: ~1.5h on H100-80GB.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-130m-hf"
LAYER=16
SEED=0
N_EDITS=500
BENCH=zsre
OUT="results/${BENCH}_shard_mamba_mamba130m_seed${SEED}.json"

mkdir -p results

if [[ -f "$OUT" ]]; then
  echo "[skip] $OUT already exists"
  exit 0
fi

echo "============================================================="
echo " STEER (SHARD-v*) / zsre / Mamba-130M (missing-scale fill-in)"
echo "============================================================="
python run_shard_mamba.py \
  --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
  --layer "$LAYER" --kind out_proj \
  --value_optim vstar \
  --v_steps 200 --v_lr 1.0 --v_weight_decay 0.0 \
  --v_norm_constraint 20.0 --sim_threshold 0.7 \
  --capture_position prompt_last --fire_position last \
  --out "$OUT"

python - <<'PY'
import json
d = json.load(open("results/zsre_shard_mamba_mamba130m_seed0.json"))
r = next((r for r in d["results"] if r["N"] == 500), None)
if r:
    print(f"zsRE / Mamba-130M / seed 0:  "
          f"Eff={r['efficacy']['accuracy']:.4f}  "
          f"Gen={r['generalization']['accuracy']:.4f}  "
          f"Spec={r['specificity']['accuracy']:.4f}")
PY
