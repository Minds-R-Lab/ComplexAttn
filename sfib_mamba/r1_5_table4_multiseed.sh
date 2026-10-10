#!/usr/bin/env bash
# r1_5_table4_multiseed.sh -- Reviewer #1 point 5 (ESWA R1.5).
#
# Reviewer #1 notes that Table 4 (STEER vs GRACE-for-Mamba head-to-head)
# and the five-preset ablation currently rely on seed 0 only, so
# "within noise" and "8.3x" statements are asserted rather than tested.
#
# This script re-runs the four Table-4 cells (STEER + GRACE on CF + zsRE
# at Mamba-790M) for seeds {1, 2} to produce a three-seed set alongside
# the already-committed seed-0 numbers. The five-preset ablation itself
# is re-run by r1_5_ablation_multiseed.sh (five presets x two cells x
# two new seeds = 20 more cells; split out for clarity).
#
# Per-cell wall time ~ 3-4h; total compute ~16h.
#
# The runner auto-skips any cell whose output JSON already exists, so
# re-running this script is safe.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="state-spaces/mamba-790m-hf"
LAYER=32
N_EDITS=500
mkdir -p results

run_shard () {
  local BENCH=$1 SEED=$2
  local OUT="results/${BENCH}_shard_mamba_mamba790m_seed${SEED}.json"
  if [[ -f "$OUT" ]]; then echo "[skip] $OUT"; return 0; fi
  echo "============================================================="
  echo " STEER (SHARD-v*) / ${BENCH} / seed ${SEED}"
  echo "============================================================="
  python run_shard_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --value_optim vstar \
    --v_steps 200 --v_lr 1.0 --v_weight_decay 0.0 \
    --v_norm_constraint 20.0 --sim_threshold 0.7 \
    --capture_position prompt_last --fire_position last \
    --out "$OUT"
}

run_grace () {
  local BENCH=$1 SEED=$2
  local OUT="results/${BENCH}_grace_mamba_mamba790m_seed${SEED}.json"
  if [[ -f "$OUT" ]]; then echo "[skip] $OUT"; return 0; fi
  echo "============================================================="
  echo " GRACE-for-Mamba / ${BENCH} / seed ${SEED}"
  echo "============================================================="
  python run_grace_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits "$N_EDITS" \
    --layer "$LAYER" --kind out_proj \
    --eps_init 1.0 --n_steps 100 \
    --out "$OUT"
}

for SEED in 1 2; do
  for BENCH in counterfact zsre; do
    run_shard "$BENCH" "$SEED"
    run_grace "$BENCH" "$SEED"
  done
done

echo
echo "============================================================="
echo " Table-4 multi-seed summary (STEER vs GRACE, N=500, Mamba-790M):"
echo "============================================================="
python - <<'PY'
import json, glob, statistics as st
def collect(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        with open(p) as fh: d = json.load(fh)
        r = next((r for r in d["results"] if r["N"] == 500), None)
        if r is None: continue
        key = "seed" + p.split("seed")[-1].split(".")[0]
        out[key] = (r["efficacy"]["accuracy"],
                    r["generalization"]["accuracy"],
                    r["specificity"]["accuracy"])
    return out

def row(label, seeds):
    xs = list(seeds.values())
    if not xs: return
    eff  = [x[0] for x in xs]
    gen  = [x[1] for x in xs]
    spec = [x[2] for x in xs]
    def ms(xs): return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)
    me, se = ms(eff); mg, sg = ms(gen); ms_, ss_ = ms(spec)
    seedlist = ",".join(sorted(seeds.keys()))
    print(f"{label:<35} n={len(xs)} ({seedlist})  "
          f"Eff={me:.4f}+-{se:.4f}  Gen={mg:.4f}+-{sg:.4f}  Spec={ms_:.4f}+-{ss_:.4f}")

for bench in ["counterfact", "zsre"]:
    row(f"STEER / {bench}", collect(f"results/{bench}_shard_mamba_mamba790m_seed*.json"))
    row(f"GRACE / {bench}", collect(f"results/{bench}_grace_mamba_mamba790m_seed*.json"))
PY
