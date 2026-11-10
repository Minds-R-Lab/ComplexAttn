#!/usr/bin/env bash
# =============================================================================
# R2 revision -- Reviewer 1 point R1.3
# Layer and module ablation for STEER on Mamba-790M + CounterFact, N = 500.
#
# The actual runner `sfib_mamba/r1_3_edit_site_ablation.sh` was written
# during R1 but never executed (it's committed, just unrun). It sweeps:
#   layer in {16, 24, 32, 40} x kind = out_proj     (4 cells)
#   layer = 32              x kind in {in_proj, x_proj}  (2 cells)
# = 6 cells, ~3.1 h each on H100-80GB (v_steps=200 is the bottleneck).
# Total wall-clock: ~19 h. Run inside tmux or nohup.
#
# All outputs land in sfib_mamba/results/edit_site_L<L>_<kind>_counterfact_mamba790m_seed0.json
# and the runner's final block prints a consolidated summary table.
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/ComplexAttn/ComplexAttn}"

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  for CAND in "$HOME/env/bin/activate" "$REPO_ROOT/.venv/bin/activate" "$HOME/ComplexAttn/env/bin/activate"; do
    if [[ -f "$CAND" ]]; then source "$CAND"; break; fi
  done
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r2_07] $(date -Iseconds) start: delegating to sfib_mamba/r1_3_edit_site_ablation.sh"
echo "[r2_07] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[r2_07] venv: ${VIRTUAL_ENV:-<none>}"

bash "$REPO_ROOT/sfib_mamba/r1_3_edit_site_ablation.sh"

echo "[r2_07] $(date -Iseconds) done"
echo
echo "Next: the 6 edit_site_L*_*.json files under sfib_mamba/results/"
echo "populate the new Section 6.6 (tab:layer-module-ablation) in the manuscript."
