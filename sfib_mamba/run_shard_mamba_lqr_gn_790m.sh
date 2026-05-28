#!/usr/bin/env bash
# run_shard_mamba_lqr_gn_790m.sh -- validation of the multi-step Gauss-Newton
# LQR variant of SHARD-for-Mamba.
#
# Replaces the one-shot LQR feedback law with N_LQR_ITERS=5 iterations of:
#   forward + backward at current delta_v -> gradient g
#   delta_v -= (gamma * ||v_orig|| / N_LQR_ITERS) * g / ||g||
#   clamp ||delta_v|| <= gamma * ||v_orig||
#
# 5 backward passes per insertion (40x cheaper than v*'s 200-step loop).
# Expected to close the Eff gap on zsRE that the one-shot LQR left open.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-790m-hf"
LAYER=32
SEED=0
N_EDITS=500

V_NORMCAP=20.0
TAU=0.7

LQR_ALPHA_SCALE=1.0
N_LQR_ITERS=5

run_cell () {
  local BENCH=$1
  local OUT="results/${BENCH}_shard_mamba_lqr_gn_mamba790m_seed${SEED}.json"
  echo "================================================================"
  echo " SHARD-LQR-GN / ${BENCH} / ${MODEL} / layer ${LAYER} / iters ${N_LQR_ITERS} / seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists."
    return 0
  fi
  python run_shard_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --value_optim lqr_gn \
    --n_lqr_iters "$N_LQR_ITERS" \
    --lqr_alpha_scale "$LQR_ALPHA_SCALE" \
    --v_norm_constraint "$V_NORMCAP" --sim_threshold "$TAU" \
    --capture_position prompt_last --fire_position last \
    --out "$OUT"
}

run_cell counterfact
run_cell zsre

echo
echo "================================================================"
echo " Mamba-790M  3-way value-optimizer comparison @ N=500:"
echo "================================================================"
python - <<'PY'
import json, os

paths = {
    ("CF",   "v*"):     "results/counterfact_shard_mamba_mamba790m_seed0.json",
    ("CF",   "lqr"):    "results/counterfact_shard_mamba_lqr_mamba790m_seed0.json",
    ("CF",   "lqr_gn"): "results/counterfact_shard_mamba_lqr_gn_mamba790m_seed0.json",
    ("zsRE", "v*"):     "results/zsre_shard_mamba_mamba790m_seed0.json",
    ("zsRE", "lqr"):    "results/zsre_shard_mamba_lqr_mamba790m_seed0.json",
    ("zsRE", "lqr_gn"): "results/zsre_shard_mamba_lqr_gn_mamba790m_seed0.json",
}

def n500(path):
    if not os.path.exists(path): return None
    with open(path) as fh: d = json.load(fh)
    r = next((r for r in d["results"] if r["N"] == 500), None)
    return (d, r)

print(f"{'cell':<6} {'optim':<8} {'Eff':>7} {'Gen':>7} {'Spec':>7} {'slots':>6} {'wall_s':>9}")
print("-" * 56)
for (cell, label), path in paths.items():
    x = n500(path)
    if x is None:
        print(f"{cell:<6} {label:<8} (no file at {path})")
        continue
    d, r = x
    if r is None:
        print(f"{cell:<6} {label:<8} (no N=500)")
        continue
    wall = d.get("wall_time_stream_s", float("nan"))
    print(f"{cell:<6} {label:<8} "
          f"{r['efficacy']['accuracy']:>7.4f} "
          f"{r['generalization']['accuracy']:>7.4f} "
          f"{r['specificity']['accuracy']:>7.4f} "
          f"{d['n_slots_final']:>6} "
          f"{wall:>9.1f}")
PY
