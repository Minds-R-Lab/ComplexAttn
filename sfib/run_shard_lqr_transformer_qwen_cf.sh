#!/usr/bin/env bash
# run_shard_lqr_transformer_qwen_cf.sh -- validate the LQR variants on the
# transformer SHARD edit site (Qwen2.5-0.5B + CounterFact).
#
# Tests the claim from the Mamba paper Section 6 that the LQR derivation is
# architecture-agnostic across the SHARD primitive: the same closed-form
# saturated bang-bang / Gauss-Newton steps should produce the same
# Pareto-frontier behaviour at the transformer MLP down-projection edit
# site that they do at Mamba's W_o.
#
# Runs three value-optimizer modes:
#   vstar   - 200-step Adam (the original SHARD-transformer v* baseline)
#   lqr     - one-shot saturated LQR step (1 backward pass)
#   lqr_gn  - multi-step Gauss-Newton LQR, n_lqr_iters=10 (10 backward passes)
#
# All on the same cell (CounterFact + Qwen2.5-0.5B, seed 0). Auto-skips
# cells whose output JSON already exists.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
LAYER=11
SEED=0
N_EDITS=500

# SHARD-transformer defaults from the paper:
V_STEPS=200
V_LR=0.5
V_WD=0.5
V_NORMCAP=4.0
TAU=0.7
N_TEMPLATES=1

LQR_ALPHA_SCALE=1.0
N_LQR_ITERS=10

run_cell () {
  local OPTIM=$1
  local OUT="results/counterfact_addressable_mem_qwen0p5b_${OPTIM}_seed${SEED}.json"
  echo "================================================================"
  echo " SHARD-transformer / CF / ${MODEL} / layer ${LAYER} / optim ${OPTIM} / seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists."
    return 0
  fi
  python run_counterfact.py \
    --method addressable_mem \
    --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --memit_layer "$LAYER" \
    --memit_v_steps "$V_STEPS" --memit_v_lr "$V_LR" \
    --mem_v_weight_decay "$V_WD" \
    --mem_v_norm_constraint "$V_NORMCAP" \
    --mem_sim_threshold "$TAU" \
    --mem_n_templates "$N_TEMPLATES" \
    --mem_value_optim "$OPTIM" \
    --mem_n_lqr_iters "$N_LQR_ITERS" \
    --mem_lqr_alpha_scale "$LQR_ALPHA_SCALE" \
    --out "$OUT"
}

run_cell vstar
run_cell lqr
run_cell lqr_gn

echo
echo "================================================================"
echo " Qwen2.5-0.5B + CounterFact  3-way value-optimizer comparison @ N=500:"
echo "================================================================"
python - <<'PY'
import json, os
paths = {
    "vstar":   "results/counterfact_addressable_mem_qwen0p5b_vstar_seed0.json",
    "lqr":     "results/counterfact_addressable_mem_qwen0p5b_lqr_seed0.json",
    "lqr_gn":  "results/counterfact_addressable_mem_qwen0p5b_lqr_gn_seed0.json",
}
def n500(path):
    if not os.path.exists(path): return None
    with open(path) as fh: d = json.load(fh)
    r = next((r for r in d.get("results", []) if r.get("N") == 500), None)
    return (d, r)
print(f"{'optim':<8} {'Eff':>8} {'Gen':>8} {'Spec':>8} {'wall_s':>10}")
print("-" * 50)
for label, path in paths.items():
    x = n500(path)
    if x is None:
        print(f"{label:<8} (no file at {path})")
        continue
    d, r = x
    if r is None:
        print(f"{label:<8} (no N=500 in {path})")
        continue
    wall = d.get("wall_time_stream_s", float("nan"))
    eff  = r['efficacy']['accuracy']
    gen  = r['generalization']['accuracy']
    spec = r['specificity']['accuracy']
    print(f"{label:<8} {eff:>8.4f} {gen:>8.4f} {spec:>8.4f} {wall:>10.1f}")
PY
