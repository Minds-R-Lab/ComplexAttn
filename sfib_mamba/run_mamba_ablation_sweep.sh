#!/usr/bin/env bash
# run_mamba_ablation_sweep.sh -- 5-preset ablation sweep on Mamba-790M.
#
# Runs the SHARD-vs-GRACE design-axis ablation on CounterFact, seed 0.
# Each preset flips one of routing / write / value-optim relative to the
# pure SHARD configuration. Closes the placeholder in paper.tex Section 5.5.
#
# Compute budget at Mamba-790M with the sequential implementation:
#   shard           ~3h    (200-step v* per insertion)
#   ablate_routing  ~3h
#   ablate_write    ~3h
#   ablate_optim    ~1.5h  (100-step Adam, no v* tricks)
#   all_grace       ~2.5h  (100-step Adam + GRACE expanding-eps)
#   ---------------
#   total          ~13h

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-790m-hf"
LAYER=32
SEED=0
N_EDITS=500
BENCH=counterfact

run_cell () {
  local PRESET=$1
  local OUT="results/${BENCH}_mamba_ablation_${PRESET}_mamba790m_seed${SEED}.json"
  echo "================================================================"
  echo " Mamba ablation / ${BENCH} / ${MODEL} / layer ${LAYER} / preset ${PRESET} / seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists."
    return 0
  fi
  python run_mamba_ablations.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --preset "$PRESET" \
    --out "$OUT"
}

for PRESET in shard ablate_routing ablate_write ablate_optim all_grace; do
  run_cell "$PRESET"
done

echo
echo "================================================================"
echo " 5-preset Mamba ablation summary @ N=500 (CF, layer 32, seed 0):"
echo "================================================================"
python - <<'PY'
import json, os
presets = ["shard", "ablate_routing", "ablate_write", "ablate_optim", "all_grace"]
print(f"{'preset':<16} {'Eff':>8} {'Gen':>8} {'Spec':>8} {'slots':>6} {'wall_s':>10}")
print("-" * 62)
for preset in presets:
    path = f"results/counterfact_mamba_ablation_{preset}_mamba790m_seed0.json"
    if not os.path.exists(path):
        print(f"{preset:<16} (no file)")
        continue
    with open(path) as fh: d = json.load(fh)
    r = next((r for r in d["results"] if r["N"] == 500), None)
    if r is None:
        print(f"{preset:<16} (no N=500)")
        continue
    wall = d.get("wall_time_stream_s", float("nan"))
    print(f"{preset:<16} "
          f"{r['efficacy']['accuracy']:>8.4f} "
          f"{r['generalization']['accuracy']:>8.4f} "
          f"{r['specificity']['accuracy']:>8.4f} "
          f"{d['n_slots_final']:>6} "
          f"{wall:>10.1f}")
PY
