#!/usr/bin/env bash
# run_grace_scaling.sh -- GRACE on the Qwen2.5 family at 1.5B / 3B / 7B on CounterFact (seed 0).
# Parallels the existing SHARD scaling sweep. Wall time on H100 ~ 25-35 min per cell.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=0
N_EDITS=500
N_STEPS=100
LR=1.0
EPS_INIT=1.0
INIT=zeros
BENCH=counterfact

run_cell () {
  local MODEL=$1 LAYER=$2 TAG=$3
  local OUT="results/${BENCH}_grace_${TAG}_seed${SEED}.json"
  echo "================================================================"
  echo " GRACE  /  ${BENCH}  /  ${MODEL}  /  layer ${LAYER}  /  seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists."
    return 0
  fi
  python run_grace.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --grace_layer "$LAYER" --grace_n_steps "$N_STEPS" --grace_lr "$LR" \
    --grace_eps_init "$EPS_INIT" --grace_init "$INIT" --out "$OUT"
}

# Layer choice per family member matches the SHARD scaling table.
run_cell Qwen/Qwen2.5-1.5B-Instruct  20  qwen1_5b
run_cell Qwen/Qwen2.5-3B-Instruct    26  qwen3b
run_cell Qwen/Qwen2.5-7B-Instruct    20  qwen7b

echo
echo "================================================================"
echo " GRACE scaling summary @ N=500:"
echo "================================================================"
python - <<'PY'
import json, os
files = [
    ("0.5B (existing)",  "results/counterfact_grace_qwen0_5b_seed0.json"),
    ("1.5B",              "results/counterfact_grace_qwen1_5b_seed0.json"),
    ("3B",                "results/counterfact_grace_qwen3b_seed0.json"),
    ("7B",                "results/counterfact_grace_qwen7b_seed0.json"),
]
def f(x): return "n/a   " if x is None else f"{x:.4f}"
print(f"{'scale':<20} {'Eff':>7} {'Gen':>7} {'Spec':>7} {'slots':>6}")
for tag, path in files:
    if not os.path.exists(path):
        print(f"{tag:<20} (missing)"); continue
    with open(path) as fh:
        d = json.load(fh)
    r500 = next((r for r in d["results"] if r["N"] == 500), None)
    if r500 is None:
        print(f"{tag:<20} (no N=500)"); continue
    print(f"{tag:<20} "
          f"{f(r500['efficacy']['accuracy']):>7} "
          f"{f(r500['generalization']['accuracy']):>7} "
          f"{f(r500['specificity']['accuracy']):>7} "
          f"{d['n_slots_final']:>6}")
PY
