#!/usr/bin/env bash
# run_shard_mamba_lqr_790m.sh -- validation of the control-theoretic LQR
# variant of SHARD-for-Mamba.
#
# The LQR variant replaces the 200-step Adam optimization of delta_v with a
# one-backward-pass closed-form Tikhonov / Gauss-Newton step:
#
#     delta_v* = - alpha * g / (||g||^2 + lambda_eff)
#
# with alpha = CE_0 * (||g||^2 + lambda_eff) / ||g||^2  (linearized-CE-to-zero
# LQR feedback). This is roughly 200x cheaper per insertion than the v*
# multi-step loop.
#
# We run on Mamba-790M (48 layers, layer 32, ~67% depth), CounterFact + zsRE,
# matched against the existing run_shard_mamba_790m.sh v* numbers.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-790m-hf"
LAYER=32
SEED=0
N_EDITS=500

V_NORMCAP=20.0
TAU=0.7

LQR_LAMBDA=1e-3
LQR_ALPHA_SCALE=1.0

run_cell () {
  local BENCH=$1
  local OUT="results/${BENCH}_shard_mamba_lqr_mamba790m_seed${SEED}.json"
  echo "================================================================"
  echo " SHARD-LQR / ${BENCH} / ${MODEL} / layer ${LAYER} / seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists."
    return 0
  fi
  python run_shard_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --value_optim lqr \
    --lqr_lambda "$LQR_LAMBDA" --lqr_alpha_scale "$LQR_ALPHA_SCALE" \
    --v_norm_constraint "$V_NORMCAP" --sim_threshold "$TAU" \
    --capture_position prompt_last --fire_position last \
    --out "$OUT"
}

run_cell counterfact
run_cell zsre

echo
echo "================================================================"
echo " Mamba-790M  SHARD-LQR  vs  SHARD-v*  @ N=500:"
echo "================================================================"
python - <<'PY'
import json, os

pairs = [
    ("CF",   "results/counterfact_shard_mamba_mamba790m_seed0.json",
              "results/counterfact_shard_mamba_lqr_mamba790m_seed0.json"),
    ("zsRE", "results/zsre_shard_mamba_mamba790m_seed0.json",
              "results/zsre_shard_mamba_lqr_mamba790m_seed0.json"),
]

def n500(path):
    if not os.path.exists(path): return None
    with open(path) as fh: d = json.load(fh)
    r = next((r for r in d["results"] if r["N"] == 500), None)
    return (d, r)

print(f"{'cell':<10} {'optim':<10} {'Eff':>8} {'Gen':>8} {'Spec':>8} {'slots':>7} {'wall_s':>10}")
print("-" * 65)
for tag, v_path, lqr_path in pairs:
    for label, path in [("v*", v_path), ("lqr", lqr_path)]:
        x = n500(path)
        if x is None:
            print(f"{tag:<10} {label:<10} (no file at {path})")
            continue
        d, r = x
        if r is None:
            print(f"{tag:<10} {label:<10} (no N=500)")
            continue
        wall = d.get("wall_time_stream_s", float("nan"))
        print(f"{tag:<10} {label:<10} "
              f"{r['efficacy']['accuracy']:>8.4f} "
              f"{r['generalization']['accuracy']:>8.4f} "
              f"{r['specificity']['accuracy']:>8.4f} "
              f"{d['n_slots_final']:>7} "
              f"{wall:>10.1f}")
PY
