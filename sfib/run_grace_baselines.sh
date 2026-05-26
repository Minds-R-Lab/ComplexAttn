#!/usr/bin/env bash
# run_grace_baselines.sh -- GRACE baseline on the four real-data cells
# (CounterFact + zsRE) x (Qwen2.5-0.5B + TinyLlama-1.1B), seed 0.
#
# Outputs JSON files under sfib/results/. The eval functions are imported
# verbatim from run_counterfact.py / run_zsre.py, so the metric definitions
# are identical to the existing AddressableMemory/MEMIT/seq_ft/frozen runs.

set -e
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=0
N_EDITS=500

# GRACE hyperparameters. The only deviation from the paper is `init=zeros`,
# because random initialization explodes the first-step loss on modern
# instruction-tuned models. Layer choices match what we used for SHARD/MEMIT.
N_STEPS=100
LR=1.0
EPS_INIT=1.0
INIT=zeros

QWEN05B_LAYER=17     # Qwen2.5-0.5B has 24 layers
TINYLLAMA_LAYER=13   # TinyLlama-1.1B-Chat-v1.0 has 22 layers


run_cell () {
  local BENCH=$1 MODEL=$2 LAYER=$3 OUT=$4
  echo "================================================================"
  echo " GRACE  /  $BENCH  /  $MODEL  /  layer $LAYER  /  seed $SEED"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists."
    return 0
  fi
  python run_grace.py \
    --benchmark "$BENCH" \
    --model "$MODEL" \
    --seed "$SEED" \
    --n_edits "$N_EDITS" \
    --grace_layer "$LAYER" \
    --grace_n_steps "$N_STEPS" \
    --grace_lr "$LR" \
    --grace_eps_init "$EPS_INIT" \
    --grace_init "$INIT" \
    --out "$OUT"
}

# ---- CounterFact ------------------------------------------------------
run_cell counterfact Qwen/Qwen2.5-0.5B-Instruct  "$QWEN05B_LAYER" \
         results/counterfact_grace_qwen0_5b_seed0.json

run_cell counterfact TinyLlama/TinyLlama-1.1B-Chat-v1.0  "$TINYLLAMA_LAYER" \
         results/counterfact_grace_tinyllama_seed0.json

# ---- zsRE -------------------------------------------------------------
run_cell zsre Qwen/Qwen2.5-0.5B-Instruct  "$QWEN05B_LAYER" \
         results/zsre_grace_qwen0_5b_seed0.json

run_cell zsre TinyLlama/TinyLlama-1.1B-Chat-v1.0  "$TINYLLAMA_LAYER" \
         results/zsre_grace_tinyllama_seed0.json

echo
echo "================================================================"
echo " All GRACE cells complete. Summary @ N=500:"
echo "================================================================"
python - <<'PY'
import json, os
files = [
    "results/counterfact_grace_qwen0_5b_seed0.json",
    "results/counterfact_grace_tinyllama_seed0.json",
    "results/zsre_grace_qwen0_5b_seed0.json",
    "results/zsre_grace_tinyllama_seed0.json",
]
def f(x):
    return "n/a   " if x is None else f"{x:.4f}"
print(f"{'cell':<55}  {'Eff':>7}  {'Gen':>7}  {'Spec':>7}  {'slots':>6}")
for path in files:
    if not os.path.exists(path):
        print(f"{path:<55}  (missing)")
        continue
    with open(path) as fh:
        d = json.load(fh)
    r500 = next((r for r in d["results"] if r["N"] == 500), None)
    if r500 is None:
        print(f"{path:<55}  (no N=500 entry)")
        continue
    print(f"{path:<55}  "
      