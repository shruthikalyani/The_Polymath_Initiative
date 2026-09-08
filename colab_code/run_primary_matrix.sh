#!/usr/bin/env bash
set -euo pipefail

# Each Python process unloads its model before the next model is loaded.
# The defaults are the three required Qwen primary-family models and the four
# core conditions. Set INCLUDE_RAW_FULL=1 only after the profile run confirms
# full conversations fit your Colab GPU. Graph traversal is optional.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS=(qwen2_5_1_5b qwen2_5_7b qwen2_5_14b)
EXPERIMENTS=(raw_truncated summary_rag facts_rag summary_tiered)
[[ "${INCLUDE_RAW_FULL:-0}" == "1" ]] && EXPERIMENTS+=(raw_full)
[[ "${INCLUDE_GRAPH:-0}" == "1" ]] && EXPERIMENTS+=(graph_traversal)

for model in "${MODELS[@]}"; do
  for experiment in "${EXPERIMENTS[@]}"; do
    python "$ROOT_DIR/colab_code/run_local_evaluation.py" \
      --model-key "$model" \
      --experiment "$experiment" \
      --data-file "$ROOT_DIR/locomo10.json" \
      --out-file "$ROOT_DIR/results/${model}__${experiment}.json" \
      --emb-dir "$ROOT_DIR/memory_cache/$model" \
      --context-token-budget 6000 \
      --top-k 5 \
      --tier-working-turns 12 \
      --log-memory-trace
  done
done
