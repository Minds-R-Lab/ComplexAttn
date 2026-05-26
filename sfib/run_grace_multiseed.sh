#!/usr/bin/env bash
# run_grace_multiseed.sh -- GRACE on the four real-data cells, seeds 1 and 2.
# Combined with the existing seed-0 runs this gives 3-seed validation matching SHARD.
# Total wall time on H100 ~ 8 cells * 20 min = ~2.5 hours.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_EDITS=500
N_STEPS=100
LR=1.0
EPS_INIT=1.0
INIT=zeros

QWEN05B_LAYER=17
TINYLLAMA_LAYER=13

run_cell () {
  local BENCH=$1 MODEL=$2 LAYER=$3 SEED=$4 TAG=$5
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

for SEED in 1 2; do
  run_cell counterfact Qwen/Qwen2.5-0.5B-Instruct        "$QWEN05B_LAYER"   "$SEED" qwen0_5b
  run_cell counterfact TinyLlama/TinyLlama-1.1B-Chat-v1.0 "$TINYLLAMA_LAYER" "$SEED" tinyllama
  run_cell zsre        Qwen/Qwen2.5-0.5B-Instruct        "$QWEN05B_LAYER"   "$SEED" qwen0_5b
  run_cell zsre        TinyLlama/TinyLlama-1.1B-Chat-v1.0 "$TINYLLAMA_LAYER" "$SEED" tinyllama
done

echo
echo "================================================================"
echo " GRACE multi-seed summary @ N=500 (mean +/- std over seeds 0,1,2):"
echo "================================================================"
python - <<'PY'
import json, os, statistics
cells = [
    ("CF / Qwen-0.5B",   "results/counterfact_grace_qwen0_5b_seed{}.json"),
    ("CF / TinyLlama",   "results/counterfact_grace_tinyllama_seed{}.json"),
    ("zsRE / Qwen-0.5B", "results/zsre_grace_qwen0_5b_seed{}.json"),
    ("zsRE / TinyLlama", "results/zsre_grace_tinyllama_seed{}.json"),
]
def f(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return "n/a"
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.4f} +/- {s:.4f}"

print(f"{'cell':<22} {'Eff@500':<19} {'Gen@500':<19} {'Spec@500':<19}")
for label, tmpl in cells:
    effs, gens, specs = [], [], []
    for seed in (0, 1, 2):
        path = tmpl.format(seed)
        if not os.path.exists(path): continue
        with open(path) as fh: d = json.load(fh)
        r500 = next((r for r in d["results"] if r["N"] == 500), None)
        if r500 is None: continue
        effs.append(r500["efficacy"]["accuracy"])
        gens.append(r500["generalization"]["accuracy"])
        specs.append(r500["specificity"]["accuracy"])
    print(f"{label:<22} {f(effs):<19} {f(gens):<19} {f(specs):<19}")
PY
