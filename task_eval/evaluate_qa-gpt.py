import os
import sys
import json
import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import eval_question_answering
from evaluation_stats import analyze_aggr_acc
from utils_eval_architectures import get_llama_answers

# Keep the architecture-aware evaluator separate from LoCoMo's legacy
# ``utils_eval.py``.  The legacy module expects --rag-mode/--use-rag/
# --access-mode flags, while this entry point uses one --experiment flag.

EXPERIMENT_CHOICES = [
    "raw_truncated",
    "raw_full",
    "summary_rag",
    "facts_rag",
    "summary_tiered",
    "graph_traversal",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoCoMo adapted memory-architecture evaluator"
    )
    parser.add_argument("--out-file", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--data-file", required=True, type=str)

    parser.add_argument(
        "--experiment",
        required=True,
        choices=EXPERIMENT_CHOICES,
        help=(
            "One controlled LoCoMo memory condition: "
            "raw_truncated, raw_full, summary_rag, facts_rag, "
            "summary_tiered, or graph_traversal."
        ),
    )
    parser.add_argument(
        "--context-token-budget",
        type=int,
        default=6000,
        help="Token budget used by raw_truncated and recorded for other conditions.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of vector-retrieved memory items."
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["contriever", "dragon"],
        default="contriever",
        help="Local query/context encoder for vector-memory conditions.",
    )
    parser.add_argument(
        "--emb-dir",
        type=str,
        default="memory_cache",
        help="Persistent per-sample memory-artifact cache directory.",
    )
    parser.add_argument(
        "--tier-working-turns",
        type=int,
        default=12,
        help="Number of most recent raw turns kept in the tiered working set.",
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="Kept for LoCoMo CLI compatibility; adapted evaluator runs QA individually.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Kept for compatibility with the original LoCoMo CLI; not used by Groq inference.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-memory-trace", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible category-5 option ordering.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openrouter", "groq"],
        default="openrouter",
        help="Chat-completions backend. Uses existing run_llama_open_router or run_llama_groq.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate only the first N LoCoMo samples.")
    parser.add_argument("--max-questions", type=int, default=None, help="Evaluate only the first N questions per sample.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    print("=" * 90)
    print(
        f"LoCoMo adapted evaluation | model={args.model} | experiment={args.experiment} | provider={args.provider}"
    )
    print("=" * 90)

    samples = json.load(open(args.data_file, encoding="utf-8"))
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    prediction_key = f"{args.model}_{args.experiment}_prediction"
    model_key = f"{args.model}_{args.experiment}"

    if os.path.exists(args.out_file):
        out_samples = {
            d["sample_id"]: d for d in json.load(open(args.out_file, encoding="utf-8"))
        }
    else:
        out_samples = {}

    for data in samples:
        sample_id = data["sample_id"]
        if sample_id in out_samples:
            out_data = out_samples[sample_id]
            # Make sure any newly added QA fields exist without destroying prior results.
            if "qa" not in out_data:
                out_data["qa"] = data["qa"].copy()
        else:
            out_data = {"sample_id": sample_id, "qa": data["qa"].copy()}

        if args.max_questions is not None:
            data = dict(data)
            data["qa"] = data["qa"][: args.max_questions]
            out_data = dict(out_data)
            out_data["qa"] = out_data["qa"][: args.max_questions]

        answers = get_llama_answers(data, out_data, prediction_key, args)

        exact_matches, lengths, recall = eval_question_answering(
            answers["qa"], prediction_key
        )

        for i in range(len(answers["qa"])):
            answers["qa"][i][model_key + "_f1"] = round(exact_matches[i], 3)
            if len(recall) > 0 and args.experiment.endswith("rag"):
                answers["qa"][i][model_key + "_recall"] = round(recall[i], 3)

        out_samples[sample_id] = answers

        with open(args.out_file, "w", encoding="utf-8") as f:
            json.dump(list(out_samples.values()), f, indent=2)

    analyze_aggr_acc(
        args.data_file,
        args.out_file,
        args.out_file.replace(".json", "_stats.json"),
        model_key,
        model_key + "_f1",
        rag=args.experiment.endswith("rag"),
    )

    print(f"Finished. Predictions: {args.out_file}")
    print(f"Stats: {args.out_file.replace('.json', '_stats.json')}")


if __name__ == "__main__":
    main()
