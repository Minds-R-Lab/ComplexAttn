#!/usr/bin/env bash
# r5_3_lqr_gn_all_scales.sh -- Reviewer #5 point 3 (ESWA R5.3).
#
# Reviewer #5 asks for LQR-GN on more Mamba scales (currently it was
# only validated on 790M + 2.8B) and for a wider trade-off view.
#
# This script completes the LQR-GN sweep by (a) adding Mamba-130M and
# (b) re-running the already-done 790M and 2.8B cells for seeds 1 and 2
# so a 3-seed set is available for each scale on CF + zsRE.
#
# Per-cell wall time at lqr_gn (n_iters=10) is 20x cheaper than v*, so:
#   130M  / CF   / seeds {0, 1, 2}  = 3 * 5 min  = 15 min
#   130M  / zsRE / seeds {0, 1, 2}  = 3 * 6 min  = 18 min
#   790M  / CF   / seeds {1, 2}     = 2 * 10 min = 20 min   (seed 0 done)
#   790M  / zsRE / seeds {1, 2}     = 2 * 14 min = 28 min
#   2.8B  / CF   / seeds {1, 2}     = 2 * 30 min = 60 min
#   2.8B  / zsRE / seeds {1, 2}     = 2 * 42 min = 84 min
#   --------------------------------------------------------
#   total                           ~ 3.75 h

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LQR_ALPHA=1.0
N_ITERS=10
mkdir -p results

run_cell () {
  local MODEL=$1 SCALE_TAG=$2 LAYER=$3 BENCH=$4 SEED=$5
  local OUT="results/${BENCH}_shard_mamba_lqr_gn${N_ITERS}_${SCALE_TAG}_seed${SEED}.json"
  if [[ -f "$OUT" ]]; then echo "[skip] $OUT"; return 0; fi
  echo "============================================================="
  echo " LQR-GN${N_ITERS} / ${BENCH} / ${SCALE_TAG} / seed ${SEED}"
  echo "============================================================="
  python run_shard_mamba.py \
    --benchmark "$BENCH" --model "$MODEL" --seed "$SEED" --n_edits 500 \
    --layer "$LAYER" --kind out_proj \
    --value_optim lqr_gn \
    --lqr_alpha_scale "$LQR_ALPHA" --n_lqr_iters "$N_ITERS" \
    --v_norm_constraint 20.0 --sim_threshold 0.7 \
    --capture_position prompt_last --fire_position last \
    --out "$OUT"
}

# Mamba-130M: all three seeds, both benchmarks (new scale)
for SEED in 0 1 2; do
  for BENCH in counterfact zsre; do
    run_cell "state-spaces/mamba-130m-hf"  mamba130m 16 "$BENCH" "$SEED"
  done
done

# Mamba-790M: seeds 1 and 2 only (seed 0 already done)
for SEED in 1 2; do
  for BENCH in counterfact zsre; do
    run_cell "state-spaces/mamba-790m-hf"  mamba790m 32 "$BENCH" "$SEED"
  done
done

# Mamba-2.8B: seeds 1 and 2 only
for SEED in 1 2; do
  for BENCH in counterfact zsre; do
    run_cell "state-spaces/mamba-2.8b-hf"  mamba2p8b 39 "$BENCH" "$SEED"
  done
done

echo
echo "============================================================="
echo " LQR-GN cross-scale summary"
echo "============================================================="
python - <<'PY'
import json, glob, statistics as st
for scale in ["mamba130m", "mamba790m", "mamba2p8b"]:
    for bench in ["counterfact", "zsre"]:
        xs = []
        for p in sorted(glob.glob(f"results/{bench}_shard_mamba_lqr_gn10_{scale}_seed*.json")):
            d = json.load(open(p))
            r = next((r for r in d["results"] if r["N"] == 500), None)
            if r is None: continue
            xs.append((r["efficacy"]["accuracy"],
                       r["generalization"]["accuracy"],
                       r["specificity"]["accuracy"]))
        if not xs: continue
        n = len(xs)
        def ms(v): return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
        me, se = ms([x[0] for x in xs])
        mg, sg = ms([x[1] for x in xs])
        mp, sp = ms([x[2] for x in xs])
        print(f"  {scale:<10} {bench:<12} n={n}  "
              f"Eff={me:.4f}+-{se:.4f}  Gen={mg:.4f}+-{sg:.4f}  Spec={mp:.4f}+-{sp:.4f}")
PY
