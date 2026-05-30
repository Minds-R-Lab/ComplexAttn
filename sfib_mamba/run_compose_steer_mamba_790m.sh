#!/usr/bin/env bash
# run_compose_steer_mamba_790m.sh -- multi-hop composition queries for STEER
# on Mamba-790M.
#
# Mines transitive chains from CounterFact (pairs of edits e1, e2 where
# target_new(e1) == subject(e2)), inserts the e1 edits via STEER, and tests
# whether the model can answer the composed two-hop query. Reports compose@N
# accuracy on both the frozen base (control) and the STEER-edited model.
#
# Total wall: ~1.5 h on H100 (50-100 chains typically; insertion via v* is
# the dominant cost). Set --max_chains 100 for a quick sanity sweep.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-790m-hf"
LAYER=32
SEED=0
N_EDITS=500
MAX_CHAINS=500

V_STEPS=200
V_LR=1.0
V_NORMCAP=20.0
TAU=0.7

OUT="results/compose_steer_mamba_mamba790m_seed${SEED}.json"

echo "================================================================"
echo " STEER composition / Mamba-790M / CounterFact transitive chains"
echo "================================================================"
if [[ -f "$OUT" ]]; then
  echo "[skip] $OUT already exists."
  exit 0
fi

python run_compose_steer_mamba.py \
  --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
  --max_chains "$MAX_CHAINS" \
  --layer "$LAYER" --kind out_proj \
  --v_steps "$V_STEPS" --v_lr "$V_LR" \
  --v_norm_constraint "$V_NORMCAP" --sim_threshold "$TAU" \
  --capture_position prompt_last --fire_position last \
  --value_optim vstar \
  --out "$OUT"

echo
echo "Results:"
python -c "
import json
d = json.load(open('$OUT'))
print(f\"  n_chains:       {d['n_chains_tested']}\")
print(f\"  n_e1_inserted:  {d['n_e1_inserted']}\")
print(f\"  base  compose:  {d['base_compose']['accuracy']:.4f}\")
print(f\"  steer compose:  {d['steer_compose']['accuracy']:.4f}\")
print(f\"  gap:            {d['compose_gap']:+.4f}\")
"
