# Colab local inference

This folder runs the existing LoCoMo memory-architecture conditions with a
locally loaded Hugging Face model. It uses deterministic 4-bit NF4 inference,
not an API. The model performs both memory management and final QA, matching
the protocol's fixed division of labor. In Colab, use
[`main.ipynb`](main.ipynb): it contains the complete cell-based setup and
execution flow.

## Model matrix

`models.json` defines the protocol's model identifiers:

- Primary scaling study: Qwen2.5 1.5B, 7B, and 14B.
- Secondary cross-family validation: Llama 3.2 3B and Llama 3.1 8B.

The two Llama repositories are gated. Before using them, accept each model's
license on Hugging Face and set `HF_TOKEN` in the Colab shell. Qwen needs no
token.

## First-time Colab setup

1. In Colab, select a GPU runtime (T4 or better). A CPU runtime cannot run the
   4-bit models.
2. Open `colab_code/main.ipynb` and run its cells in order: dependency setup,
   GPU check, optional Hugging Face authentication, then the smoke test.
3. Use the notebook's single-cell or primary-matrix cells for the actual runs.

## Run one experimental cell

Keep the same settings for every model in a comparison row. In particular, do
not lower the 6,000-token sliding-window budget for a smaller model simply to
make it fit; it would be a different experimental condition.

```bash
python colab_code/run_local_evaluation.py \
  --model-key qwen2_5_7b \
  --experiment summary_rag \
  --out-file results/qwen2_5_7b__summary_rag.json \
  --emb-dir memory_cache/qwen2_5_7b \
  --context-token-budget 6000 \
  --top-k 5 \
  --tier-working-turns 12 \
  --log-memory-trace
```

Memory artifacts are isolated by a versioned provider/model/retriever cache key
under `memory_cache/<model-key>/`.
That matters because the protocol requires each small model to generate its
own summaries, facts, and graphs rather than receiving memory written by a
larger model.

## Run the primary matrix

```bash
bash colab_code/run_primary_matrix.sh
```

This runs the core required conditions for the Qwen primary family. Add
`INCLUDE_RAW_FULL=1` only after verifying the full conversation fits the
runtime without out-of-memory errors. Add `INCLUDE_GRAPH=1` for the optional
graph condition:

```bash
INCLUDE_RAW_FULL=1 INCLUDE_GRAPH=1 bash colab_code/run_primary_matrix.sh
```

## Reproducibility notes

- Inference is greedy (`do_sample=False`) with a fixed seed of 42.
- Quantization is 4-bit NF4 with double quantization. Record this in the
  methods section; it is held constant across model cells.
- The context budget is measured with the existing fixed `cl100k_base`
  tokenizer, preserving the benchmark's current sliding-window definition.
- A `local_run` block is saved with each completed sample. Output files are
  checkpointed after each sample, so completed samples are retained after a
  Colab interruption.
- `raw_full` is valid only when the whole prompt fits. Do not silently trim it;
  use `raw_truncated` for the fixed-window baseline instead.
