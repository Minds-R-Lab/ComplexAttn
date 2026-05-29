#!/usr/bin/env bash
# run_shard_lqr_transformer_qwen_zsre.sh -- replicate the transformer LQR
# finding on a second cell (zsRE + Qwen2.5-0.5B).
#
# Mirrors run_shard_lqr_transformer_qwen_cf.sh on the zsRE benchmark.
# Strengthens the "architecture-agnosticism of LQR" claim from
# 1 transformer cell (CF + Qwen2.5-0.5B) to 2 transformer cells.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
LAYER=11
SEED=0
N_EDITS=500

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
  local OUT="results/zsre_addressable_mem_qwen0p5b_${OPTIM}_seed${SEED}.json"
  echo "================================================================"
  echo " SHARD-transformer / zsRE / ${MODEL} / layer ${LAYER} / optim ${OPTIM} / seed ${SEED}"
  echo "================================================================"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists."
    return 0
  fi
  python run_zsre.py \
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
echo " Qwen2.5-0.5B + zsRE  3-way value-optimizer comparison @ N=500:"
echo "================================================================"
python - <<'PY'
import json, os
paths = {
    "vstar":   "results/zsre_addressable_mem_qwen0p5b_vstar_seed0.json",
    "lqr":     "results/zsre_addressable_mem_qwen0p5b_lqr_seed0.json",
    "lqr_gn":  "results/zsre_addressable_mem_qwen0p5b_lqr_gn_seed0.json",
}
def find_n500(d):
    # Try multiple possible keys in case the transformer runner saves with a different name
    for r in d.get("results", []) or d.get("history", []) or []:
        for key in ("N", "n", "n_inserted", "step"):
            if r.get(key) == 500:
                return r
    return None
print(f"{'optim':<8} {'Eff':>8} {'Gen':>8} {'Spec':>8} {'wall_s':>10}")
print("-" * 50)
for label, path in paths.items():
    if not os.path.exists(path):
        print(f"{label:<8} (no file at {path})")
        continue
    with open(path) as fh: d = json.load(fh)
    r = find_n500(d)
    if r is None:
        # Last-resort: print top-level keys to help debug
        top_keys = list(d.keys())
        print(f"{label:<8} (no N=500; top keys: {top_keys})")
        continue
    wall = d.get("wall_time_stream_s", d.get("wall_time_s", float("nan")))
    eff  = r['efficacy']['accuracy']
    gen  = r['generalization']['accuracy']
    spec = r['specificity']['accuracy']
    print(f"{label:<8} {eff:>8.4f} {gen:>8.4f} {spec:>8.4f} {wall:>10.1f}")
PY
