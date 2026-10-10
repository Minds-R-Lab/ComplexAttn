#!/usr/bin/env bash
# run_eswa_revision_all.sh -- Orchestrator for the ESWA-D-26-23857
# revision experiments. Runs the cheap ones first so numbers can be
# folded into the response letter early; the expensive ones (long-stream
# to 10k edits, sequentialised ROMBA, ablation 3-seed re-run) land last.
#
# Each sub-script auto-skips any cell whose output JSON already exists,
# so re-running this is safe. Nothing destructive.
#
# Total wall-time budget on a single H100-80GB: ~70 h.
# Expect to run this overnight several times, or split across machines.

set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p results

run_phase () {
  local LABEL=$1 CMD=$2
  echo
  echo "################################################################"
  echo "  PHASE: ${LABEL}"
  echo "################################################################"
  bash -c "$CMD" || echo "  [warning] ${LABEL} returned non-zero -- continuing"
}

# -------- Tier A: cheap, hours each --------

run_phase "R4.7 -- fill in missing zsRE scale (Mamba-130M)" \
  "bash r4_7_zsre_mamba130m.sh"

run_phase "R5.3 -- LQR-GN across all three Mamba scales" \
  "bash r5_3_lqr_gn_all_scales.sh"

# -------- Tier B: moderate --------

run_phase "R1.3 -- edit-site ablation (layer x kind sweep)" \
  "bash r1_3_edit_site_ablation.sh"

run_phase "R1.4 -- tau and gamma Pareto sweep" \
  "bash r1_4_tau_gamma_sweep.sh"

run_phase "R1.5 / R4.7 -- Table 4 multi-seed completion" \
  "bash r1_5_table4_multiseed.sh"

# -------- Tier C: expensive --------

run_phase "R1.2 -- MQuAKE multi-hop evaluation (STEER)" \
  "python r1_2_mquake_eval.py \
     --model state-spaces/mamba-790m-hf --layer 32 \
     --n_items 500 --seed 0 \
     --out results/mquake_steer_mamba790m_seed0.json"

run_phase "R5.2 -- long-stream to 10k + mem/latency profile" \
  "python r5_2_long_stream_memlat.py \
     --model state-spaces/mamba-790m-hf --layer 32 \
     --n_edits 10000 --seed 0 \
     --out results/long_stream_mamba790m.json \
     --mem_log results/long_stream_mamba790m_memlog.csv"

run_phase "R1.1 -- sequentialised ROMBA baseline (needs romba_ref/)" \
  "python r1_1_romba_sequentialised.py \
     --benchmark counterfact --model state-spaces/mamba-790m-hf \
     --layer 32 --seed 0 --n_edits 500 \
     --out results/counterfact_romba_seq_mamba790m_seed0.json"

echo
echo "================================================================"
echo "  All ESWA revision phases completed (check [warning] lines above)."
echo "================================================================"
ls -la results/ | grep -E "edit_site|tau_sweep|gamma_sweep|mquake|long_stream|romba_seq|lqr_gn10|mamba130m" | head -40
