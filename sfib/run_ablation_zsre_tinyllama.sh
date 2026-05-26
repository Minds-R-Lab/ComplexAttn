#!/usr/bin/env bash
# run_ablation_zsre_tinyllama.sh -- run the 5 ablation presets on
# zsRE + TinyLlama-1.1B-Chat-v1.0 (seed 0).
#
# Companion to run_ablation_sweep.sh (which targets CounterFact + Qwen-0.5B).
# Two cells of ablation lets us confirm whether the cosine-routing finding
# generalizes -- if the same single-axis flip collapses Gen here as well,
# the paper's headline ablation claim becomes substantially stronger.
#
# Wall time estimate on H100: ~5 cells * ~35 min = ~3 hours.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LAYER=13
SEED=0
N_EDITS=500
BENCH=zsre

# SwiGLU-tuned hyperparameters (same as SHARD on TinyLlama).
V_STEPS=200
V_LR=1.0
V_WD=0.0
V_NORMCAP=20.0
TAU=0.7
EPS_INIT=1.0

run_preset () {
  local PRESET=$1
  local OUT="results/${BENCH}_abl_${PRESET}_tinyllama_seed${SEED}.json"
  echo "================================================================"
  echo " Ablation [${PRESET}]  /  ${BENCH}  /  ${MODEL}  /  layer ${LAYER}  /  seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists."
    return 0
  fi
  python run_ablations.py \
    --benchmark "$BENCH" --model "$MODEL" --layer "$LAYER" --seed "$SEED" \
    --n_edits "$N_EDITS" --preset "$PRESET" \
    --v_steps "$V_STEPS" --v_lr "$V_LR" --v_weight_decay "$V_WD" \
    --v_norm_constraint "$V_NORMCAP" --sim_threshold "$TAU" --eps_init "$EPS_INIT" \
    --out "$OUT"
}

# Run all five presets in order.
run_preset shard
run_preset ablate_routing
run_preset ablate_write
run_preset ablate_optim
run_preset all_grace

echo
echo "================================================================"
echo " Ablation sweep summary @ N=500 (zsRE + TinyLlama):"
echo "================================================================"
python - <<'PY'
import json, os
files = [
    ("shard",          "results/zsre_abl_shard_tinyllama_seed0.json"),
    ("ablate_routing", "results/zsre_abl_ablate_routing_tinyllama_seed0.json"),
    ("ablate_write",   "results/zsre_abl_ablate_write_tinyllama_seed0.json"),
    ("ablate_optim",   "results/zsre_abl_ablate_optim_tinyllama_seed0.json"),
    ("all_grace",      "results/zsre_abl_all_grace_tinyllama_seed0.json"),
]
def f(x): return "n/a   " if x is None else f"{x:.4f}"
print(f"{'preset':<18} {'routing':<10} {'write':<14} {'optim':<12} {'Eff':>7} {'Gen':>7} {'Spec':>7} {'slots':>6}")
for preset, path in files:
    if not os.path.exists(path):
        print(f"{preset:<18} (missing)"); continue
    with open(path) as fh:
        d = json.load(fh)
    r500 = next((r for r in d["results"] if r["N"] == 500), None)
    if r500 is None:
        print(f"{preset:<18} (no N=500)"); continue
    print(f"{preset:<18} {d['routing']:<10} {d['write_mode']:<14} {d['value_optim']:<12} "
          f"{f(r500['efficacy']['accuracy']):>7} {f(r500['generalization']['accuracy']):>7} "
          f"{f(r500['specificity']['accuracy']):>7} {d['n_slots_final']:>6}")
PY
