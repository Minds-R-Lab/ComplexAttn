#!/usr/bin/env bash
# run_ablation_sweep.sh -- run the five ablation presets on CounterFact + Qwen2.5-0.5B (seed 0).
# This is the smoke-test cell. If the results are interesting, run on the other three cells too.
#
# Total wall time ~ 5 cells * 20 min = ~1.5-2 hours on H100.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=Qwen/Qwen2.5-0.5B-Instruct
LAYER=17
SEED=0
N_EDITS=500
BENCH=counterfact

# SwiGLU-tuned hyperparameters (match the existing SHARD numbers).
V_STEPS=200
V_LR=1.0
V_WD=0.0
V_NORMCAP=20.0
TAU=0.7
EPS_INIT=1.0

run_preset () {
  local PRESET=$1
  local OUT="results/${BENCH}_abl_${PRESET}_qwen0_5b_seed${SEED}.json"
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
echo " Ablation sweep summary @ N=500:"
echo "================================================================"
python - <<'PY'
import json, os
files = [
    ("shard",          "results/counterfact_abl_shard_qwen0_5b_seed0.json"),
    ("ablate_routing", "results/counterfact_abl_ablate_routing_qwen0_5b_seed0.json"),
    ("ablate_write",   "results/counterfact_abl_ablate_write_qwen0_5b_seed0.json"),
    ("ablate_optim",   "results/counterfact_abl_ablate_optim_qwen0_5b_seed0.json"),
    ("all_grace",      "results/counterfact_abl_all_grace_qwen0_5b_seed0.json"),
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
