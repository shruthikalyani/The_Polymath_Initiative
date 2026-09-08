"""Run one LoCoMo memory-architecture cell locally on a Colab GPU.

The existing architecture implementation is reused unchanged. Its two remote
generation call sites are redirected to a single deterministic, 4-bit local
Transformers model, so the answering model also performs memory management.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_EVAL = ROOT / "task_eval"
sys.path.insert(0, str(TASK_EVAL))

from local_inference import LocalGenerator


EXPERIMENTS = [
    "raw_truncated",
    "raw_full",
    "summary_rag",
    "facts_rag",
    "summary_tiered",
    "graph_traversal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local 4-bit LoCoMo architecture evaluation")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-key", help="Key in colab_code/models.json")
    model_group.add_argument("--model-id", help="Explicit Hugging Face model ID")
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--data-file", default=str(ROOT / "locomo10.json"))
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--models-file", default=str(Path(__file__).with_name("models.json")))
    parser.add_argument("--context-token-budget", type=int, default=6000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever", choices=["contriever", "dragon"], default="contriever")
    parser.add_argument("--tier-working-turns", type=int, default=12)
    parser.add_argument("--emb-dir", default=str(ROOT / "memory_cache"))
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-memory-trace", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test only; omit for a full run.")
    parser.add_argument("--no-4bit", action="store_true", help="Only use if the GPU has enough VRAM for FP16.")
    return parser.parse_args()


def resolve_model(args: argparse.Namespace) -> tuple[str, str, dict]:
    if args.model_id:
        label = args.model_id.replace("/", "_").replace("-", "_").lower()
        return args.model_id, label, {"model_id": args.model_id, "role": "custom"}
    models = json.loads(Path(args.models_file).read_text(encoding="utf-8"))
    if args.model_key not in models:
        raise ValueError(f"Unknown --model-key {args.model_key!r}; choose one of {', '.join(models)}")
    return models[args.model_key]["model_id"], args.model_key, models[args.model_key]


def main() -> None:
    args = parse_args()
    model_id, model_label, model_meta = resolve_model(args)
    random.seed(args.seed)

    # Reuse the common evaluator through its explicit local-provider hook.
    # The same callable is used for memory writing and final QA.
    import utils_eval_architectures
    import memory_architectures
    from evaluation import eval_question_answering
    from evaluation_stats import analyze_aggr_acc

    # These values are used by the reused evaluator. ``model`` is deliberately
    # a stable experiment label rather than an API model slug.
    args.model = model_label
    args.batch_size = 1
    args.use_4bit = not args.no_4bit
    args.local_model_id = model_id
    args.provider = "local"
    args.llm_fn = generator.generate


    print("=" * 90)
    print(f"Local LoCoMo | model={model_id} | experiment={args.experiment} | 4bit={not args.no_4bit}")
    print("=" * 90)

    data_path = Path(args.data_file)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    out_samples = {
        item["sample_id"]: item
        for item in json.loads(out_path.read_text(encoding="utf-8"))
    } if out_path.exists() else {}

    prediction_key = f"{model_label}_{args.experiment}_prediction"
    model_key = f"{model_label}_{args.experiment}"
    for data in samples:
        sample_id = data["sample_id"]
        out_data = out_samples.get(sample_id, {"sample_id": sample_id, "qa": data["qa"].copy()})
        if "qa" not in out_data:
            out_data["qa"] = data["qa"].copy()
        answers = utils_eval_architectures.get_llama_answers(data, out_data, prediction_key, args)
        exact_matches, _lengths, recall = eval_question_answering(answers["qa"], prediction_key)
        for index, score in enumerate(exact_matches):
            answers["qa"][index][f"{model_key}_f1"] = round(score, 3)
            if args.experiment.endswith("rag") and recall:
                answers["qa"][index][f"{model_key}_recall"] = round(recall[index], 3)
        answers["local_run"] = {
            "model_id": model_id,
            "model_label": model_label,
            "model_metadata": model_meta,
            "experiment": args.experiment,
            "context_token_budget": args.context_token_budget,
            "top_k": args.top_k,
            "tier_working_turns": args.tier_working_turns,
            "retriever": args.retriever,
            "quantization": "4bit-nf4" if not args.no_4bit else "fp16",
            "seed": args.seed,
        }
        out_samples[sample_id] = answers
        out_path.write_text(json.dumps(list(out_samples.values()), indent=2), encoding="utf-8")

    stats_path = out_path.with_name(f"{out_path.stem}_stats.json")
    analyze_aggr_acc(str(data_path), str(out_path), str(stats_path), model_key, f"{model_key}_f1", rag=args.experiment.endswith("rag"))
    generator.close()
    print(f"Finished. Predictions: {out_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
