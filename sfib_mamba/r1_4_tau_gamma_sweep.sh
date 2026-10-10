#!/usr/bin/env bash
# r1_4_tau_gamma_sweep.sh -- Reviewer #1 point 4 (ESWA R1.4).
#
# Reviewer #1 asks for a sensitivity analysis on tau (firing threshold)
# and gamma (box bound / v_norm_constraint), reporting how the two
# hyperparameters move the Gen--Spec Pareto frontier.
#
# We sweep both on Mamba-790M + CounterFact (the headline cell), seed 0,
# layer 32, out_proj. tau sweep is 9-point, gamma sweep is 4-point;
# total 13 cells (9 + 4, not 36 -- we vary one at a time around the
# published defaults tau=0.7, gamma=20 so the Pareto is anchored on the
# published operating point).
#
# Per-cell wall time at the Mamba-790M scale: ~3.1 h on H100-80GB
# (v_steps=200 insertion sweep is the bottleneck).
# Total: ~13 * 3.1h = ~40 h -- consider running as nohup overnight.
#
# Outputs: results/tau_sweep_{tau}_cf_mamba790m_seed0.json
#          results/gamma_sweep_{gamma}_cf_mamba790m_seed0.json
# The r1_4 analysis then reads these and plots the Pareto curves.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-790m-hf"
LAYER=32
SEED=0
N_EDITS=500
BENCH=counterfact
V_STEPS=200
V_LR=1.0

mkdir -p results

run_tau () {
  local TAU=$1
  local OUT="results/tau_sweep_${TAU}_${BENCH}_mamba790m_seed${SEED}.json"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT"
    return 0
  fi
  echo "============================================================="
  echo " tau=${TAU}  gamma=20 (default)"
  echo "============================================================="
  python run_shard_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --value_optim vstar \
    --v_steps "$V_STEPS" --v_lr "$V_LR" --v_weight_decay 0.0 \
    --v_norm_constraint 20.0 --sim_threshold "$TAU" \
    --capture_position prompt_last --fire_position last \
    --out "$OUT"
}

run_gamma () {
  local GAMMA=$1
  local OUT="results/gamma_sweep_${GAMMA}_${BENCH}_mamba790m_seed${SEED}.json"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT"
    return 0
  fi
  echo "============================================================="
  echo " tau=0.7 (default)  gamma=${GAMMA}"
  echo "============================================================="
  python run_shard_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --value_optim vstar \
    --v_steps "$V_STEPS" --v_lr "$V_LR" --v_weight_decay 0.0 \
    --v_norm_constraint "$GAMMA" --sim_threshold 0.7 \
    --capture_position prompt_last --fire_position last \
    --out "$OUT"
}

# tau sweep around the published 0.7
for TAU in 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90; do
  run_tau "$TAU"
done

# gamma sweep around the published 20
for GAMMA in 5 10 20 40; do
  run_gamma "$GAMMA"
done

echo
echo "============================================================="
echo " tau / gamma sweep summary at N=500 (CF, Mamba-790M, seed 0):"
echo "============================================================="
python - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("results/tau_sweep_*_counterfact_mamba790m_seed0.json")):
    with open(p) as fh: d = json.load(fh)
    r500 = next((r for r in d["results"] if r["N"] == 500), None)
    if r500 is None: continue
    tag = os.path.basename(p).replace("tau_sweep_", "").replace("_counterfact_mamba790m_seed0.json", "")
    rows.append(("tau", tag, r500))
for p in sorted(glob.glob("results/gamma_sweep_*_counterfact_mamba790m_seed0.json")):
    with open(p) as fh: d = json.load(fh)
    r500 = next((r for r in d["results"] if r["N"] == 500), None)
    if r500 is None: continue
    tag = os.path.basename(p).replace("gamma_sweep_", "").replace("_counterfact_mamba790m_seed0.json", "")
    rows.append(("gamma", tag, r500))
print(f"{'kind':<6} {'val':<6} {'Eff':>8} {'Gen':>8} {'Spec':>8}")
for kind, tag, r in rows:
    print(f"{kind:<6} {tag:<6} {r['efficacy']['accuracy']:>8.4f} "
          f"{r['generalization']['accuracy']:>8.4f} {r['specificity']['accuracy']:>8.4f}")
PY
