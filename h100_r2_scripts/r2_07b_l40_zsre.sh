#!/usr/bin/env bash
# =============================================================================
# R2 revision -- Reviewer 1 point R1.3 (mitigation)
# One additional cell to accompany the layer/module ablation (r2_07):
#   STEER at layer 40, W_o, zsRE, Mamba-790M, seed 0, N = 500.
#
# The CF counterpart is already produced by r2_07 as
#   sfib_mamba/results/edit_site_L40_out_proj_counterfact_mamba790m_seed0.json
# (Eff=1.000, Gen=0.276, Spec=0.172). This script produces the zsRE counterpart
# so Section 6.6's "A separate zsRE cell at the same layer 40 reproduces the
# pattern" sentence can be filled in with real numbers rather than left as
# [pending].
#
# Wall-clock: ~3 hours on one H100-80GB (zsRE eval harness is smaller than CF's).
# Output: sfib_mamba/results/edit_site_L40_out_proj_zsre_mamba790m_seed0.json
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/ComplexAttn/ComplexAttn}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  for CAND in "$HOME/env/bin/activate" "$REPO_ROOT/.venv/bin/activate" "$HOME/ComplexAttn/env/bin/activate"; do
    if [[ -f "$CAND" ]]; then source "$CAND"; break; fi
  done
fi

cd "$REPO_ROOT/sfib_mamba"
mkdir -p results

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="results/edit_site_L40_out_proj_zsre_mamba790m_seed0.json"
if [[ -f "$OUT" ]]; then
  echo "[r2_07b] skip (exists): $OUT"
  exit 0
fi

echo "[r2_07b] $(date -Iseconds) start: STEER L40 out_proj on zsRE + Mamba-790M"
echo "[r2_07b] venv: ${VIRTUAL_ENV:-<none>}"
echo "[r2_07b] cwd:  $(pwd)"
echo "[r2_07b] out:  $OUT"

python -u run_shard_mamba.py \
  --benchmark zsre --model "state-spaces/mamba-790m-hf" --seed 0 --n_edits 500 \
  --layer 40 --kind out_proj \
  --value_optim vstar \
  --v_steps 200 --v_lr 1.0 --v_weight_decay 0.0 \
  --v_norm_constraint 20.0 --sim_threshold 0.7 \
  --capture_position prompt_last --fire_position last \
  --out "$OUT"

echo "[r2_07b] $(date -Iseconds) done"
echo
echo "Result at N=500:"
python - <<PY
import json
d = json.load(open("$OUT"))
r = next((r for r in d["results"] if r.get("N") == 500), None)
if r:
    print(f"  Eff={r['efficacy']['accuracy']:.4f}  "
          f"Gen={r['generalization']['accuracy']:.4f}  "
          f"Spec={r['specificity']['accuracy']:.4f}")
PY
echo
echo "Next: patch manuscript.tex Section 6.6, replacing"
echo "'[zsRE L40 cell pending; completes before camera-ready]'"
echo "with the Eff/Gen/Spec triple above."
