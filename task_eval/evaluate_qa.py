import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
import argparse
from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_aggr_acc
from task_eval.utils_eval_architectures import get_llama_answers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument(
        "--rag-mode",
        type=str,
        default="",
        choices=["", "summary_rag", "facts_rag"],
    )
    parser.add_argument(
        "--access-mode",
        required=True,
        choices=["truncated", "full"],
        help=(
            "Axis B (Access) for the memory-architecture study: 'truncated' "
            "applies a real tiktoken-backed sliding window over the raw "
            "conversation turns (Row 1, baseline, all models); 'full' sends "
            "the entire conversation with no truncation (Row 2, baseline, "
            "all models -- only feasible for models/tiers whose context "
            "window and TPM budget can actually fit it)."
        ),
    )
    parser.add_argument(
        "--context-token-budget",
        type=int,
        default=6000,
        help=(
            "Token budget for the conversation context when --access-mode "
            "is 'truncated' (ignored when 'full'). Leave headroom under "
            "your provider's TPM limit for prompt overhead + generation on "
            "top of this. Record whatever value you use here in your "
            "methods section -- it's a real experimental parameter now, "
            "not an implementation detail."
        ),
    )
    parser.add_argument("--use-rag", action="store_true")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--batch-size", default=1, type=int)

    parser.add_argument("--emb-dir", type=str, default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return args


def main():
    # get arguments
    args = parse_args()

    print(
        "******************  Evaluating Model %s [access_mode=%s] ***************"
        % (args.model, args.access_mode)
    )

    # load conversations
    samples = json.load(open(args.data_file))

    # access_mode is folded into both keys so Row 1 (truncated) and Row 2
    # (full) results for the *same* model land in different columns in the
    # output JSON instead of one silently overwriting the other.
    prediction_key = (
        "%s_%s_prediction" % (args.model, args.access_mode)
        if not args.use_rag
        else "%s_%s_%s_top_%s_prediction"
        % (args.model, args.access_mode, args.rag_mode, args.top_k)
    )
    model_key = (
        "%s_%s" % (args.model, args.access_mode)
        if not args.use_rag
        else "%s_%s_%s_top_%s"
        % (args.model, args.access_mode, args.rag_mode, args.top_k)
    )
    # load the output file if it exists to check for overwriting
    if os.path.exists(args.out_file):
        out_samples = {d["sample_id"]: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}

    for data in samples:
        out_data = {"sample_id": data["sample_id"]}
        if data["sample_id"] in out_samples:
            out_data["qa"] = out_samples[data["sample_id"]]["qa"].copy()
        else:
            out_data["qa"] = data["qa"].copy()

        answers = get_llama_answers(data, out_data, prediction_key, args)

        # evaluate individual QA samples and save the score
        exact_matches, lengths, recall = eval_question_answering(
            answers["qa"], prediction_key
        )
        for i in range(0, len(answers["qa"])):
            answers["qa"][i][model_key + "_f1"] = round(exact_matches[i], 3)
            if args.use_rag and len(recall) > 0:
                answers["qa"][i][model_key + "_recall"] = round(recall[i], 3)

        out_samples[data["sample_id"]] = answers

    with open(args.out_file, "w") as f:
        json.dump(list(out_samples.values()), f, indent=2)

    analyze_aggr_acc(
        args.data_file,
        args.out_file,
        args.out_file.replace(".json", "_stats.json"),
        model_key,
        model_key + "_f1",
        rag=args.use_rag,
    )
    # encoder=tiktoken.encoding_for_model(args.model))


main()
