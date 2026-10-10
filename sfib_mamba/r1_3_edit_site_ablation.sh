#!/usr/bin/env bash
# r1_3_edit_site_ablation.sh -- Reviewer #1 point 3 (ESWA R1.3).
#
# Reviewer #1 asks for sensitivity to the chosen edit site -- layer
# (currently fixed at ~2/3 depth per ROMBA), and module choice (gate
# path / in_proj vs the default out_proj).
#
# Mamba-790M has 48 layers; default is layer 32 (~67%). We sweep:
#   layer  in {16, 24, 32, 40}  (early, lower-mid, mid, upper)   x  out_proj
#   kind   in {out_proj, in_proj, x_proj}                        x  layer 32
#
# This is 4 + 2 = 6 new cells (layer=32,kind=out_proj is the already-
# committed baseline). Per-cell wall time ~3.1h; total ~19h.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-790m-hf"
SEED=0
N_EDITS=500
BENCH=counterfact

mkdir -p results

run_cell () {
  local LAYER=$1 KIND=$2
  local OUT="results/edit_site_L${LAYER}_${KIND}_${BENCH}_mamba790m_seed${SEED}.json"
  if [[ -f "$OUT" ]]; then echo "[skip] $OUT"; return 0; fi
  echo "============================================================="
  echo " edit-site ablation / layer=${LAYER} / kind=${KIND}"
  echo "============================================================="
  python run_shard_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind "$KIND" \
    --value_optim vstar \
    --v_steps 200 --v_lr 1.0 --v_weight_decay 0.0 \
    --v_norm_constraint 20.0 --sim_threshold 0.7 \
    --capture_position prompt_last --fire_position last \
    --out "$OUT"
}

# Layer sweep at kind=out_proj (the published default kind).
for LAYER in 16 24 32 40; do
  run_cell "$LAYER" out_proj
done

# Kind sweep at layer=32 (the published default layer).
for KIND in in_proj x_proj; do
  run_cell 32 "$KIND"
done

echo
echo "============================================================="
echo " edit-site ablation summary at N=500 (CF, Mamba-790M, seed 0):"
echo "============================================================="
python - <<'PY'
import json, glob, os
for p in sorted(glob.glob("results/edit_site_*_counterfact_mamba790m_seed0.json")):
    d = json.load(open(p))
    r = next((r for r in d["results"] if r["N"] == 500), None)
    tag = os.path.basename(p).replace("edit_site_", "").replace("_counterfact_mamba790m_seed0.json", "")
    if r:
        print(f"{tag:<20}  Eff={r['efficacy']['accuracy']:.4f}  "
              f"Gen={r['generalization']['accuracy']:.4f}  "
              f"Spec={r['specificity']['accuracy']:.4f}")
PY
