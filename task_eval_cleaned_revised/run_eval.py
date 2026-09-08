import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from tqdm import tqdm

from api_client import resolve_generator
from qa_engine import generate_answers
from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_aggr_acc

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_eval")


NOT_MENTIONED = "not mentioned in the conversation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long-context QA benchmark evaluator")
    p.add_argument("--out-file", required=True, help="Incremental JSON output path")

    p.add_argument("--model", required=True, help="Model identifier")
    p.add_argument("--data-file", required=True, help="Input data JSON")

    p.add_argument(
        "--provider",
        default="groq",
        choices=["groq", "openrouter", "nvidia"],
        help="Inference backend",
    )
    p.add_argument(
        "--access-mode",
        required=True,
        choices=["truncated", "full"],
        help="Axis B: truncated sliding-window vs. full context",
    )
    p.add_argument(
        "--context-token-budget",
        type=int,
        default=6000,
        help="Token budget when access-mode=truncated",
    )
    p.add_argument("--batch-size", type=int, default=1, help="Questions per LLM call")
    p.add_argument(
        "--overwrite", action="store_true", help="Re-generate existing predictions"
    )

    p.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Debug cap on QA items per sample",
    )

    p.add_argument(
        "--rag-mode",
        type=str,
        default="",
        choices=["", "dialog", "observation", "summary"],
    )
    p.add_argument("--use-rag", action="store_true")

    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--retriever", type=str, default="contriever")
    return p.parse_args()


def load_existing(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded %d existing samples from %s", len(data), path)
        return {d["sample_id"]: d for d in data}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s (%s). Starting fresh.", path, exc)
        return {}


def atomic_write(path: str, samples_map: Dict[str, dict]) -> None:

    tmp = path + ".tmp"
    for attempt in range(5):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(list(samples_map.values()), f, indent=2)
            os.replace(tmp, path)
            return
        except PermissionError:
            logger.warning(
                "File %s is locked (attempt %d/5). Retrying in 1 second...",
                path,
                attempt + 1,
            )
            time.sleep(1)
    logger.error("Could not write to %s after 5 attempts.", path)


def build_keys(model: str, access_mode: str, use_rag: bool, rag_mode: str, top_k: int):
    if use_rag:
        pred = f"{model}_{access_mode}_{rag_mode}_top_{top_k}_prediction"
        mk = f"{model}_{access_mode}_{rag_mode}_top_{top_k}"
    else:
        pred = f"{model}_{access_mode}_prediction"
        mk = f"{model}_{access_mode}"
    return pred, mk


def find_cat5_abstention_rows(qa_list: List[dict]) -> Set[int]:
    """Indices of category-5 rows with no genuine ground truth.

    These are the entity-swapped adversarial questions. A row qualifies if it
    never had an 'answer' (fresh dataset) or if a previous run stamped the
    synthetic 'MISSING_GROUND_TRUTH' placeholder onto it.
    """
    return {
        i
        for i, qa in enumerate(qa_list)
        if qa.get("category") == 5
        and qa.get("answer") in (None, "MISSING_GROUND_TRUTH")
    }


def main() -> None:
    args = parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        logger.info("Created output folder: %s", out_dir)

    # Bind provider
    generator = resolve_generator(args.provider)
    args.llm_fn = lambda query, max_tokens: generator(
        query, max_tokens, model=args.model
    )

    logger.info(
        "Evaluating model=%s provider=%s access=%s out=%s",
        args.model,
        args.provider,
        args.access_mode,
        args.out_file,
    )

    with open(args.data_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    out_samples = load_existing(args.out_file)
    prediction_key, model_key = build_keys(
        args.model, args.access_mode, args.use_rag, args.rag_mode, args.top_k
    )

    for data in tqdm(samples, desc="Samples"):
        sid = data["sample_id"]

        if sid in out_samples:
            out_data = out_samples[sid]
        else:
            out_data = {"sample_id": sid, "qa": data["qa"].copy()}

        # 1. generate answers for the taken sample
        out_data = generate_answers(data, out_data, prediction_key, args)

        # backfill any questions that were skipped (like malformed cat-5 entires)
        for qa in out_data["qa"]:
            if prediction_key not in qa:
                qa[prediction_key] = "NO_PREDICTION"

        # 2. FLUSH RAW PREDICTIONS IMMEDIATELY BEFORE DOING ANY EVAL
        out_samples[sid] = out_data
        atomic_write(args.out_file, out_samples)
        logger.info("Sample %s raw predictions flushed.", sid)

        # 3. Evaluate the hell out of it.
        try:
            # Category-5 abstention rows (entity swaps) have no ground-truth answer in the conversation. Tag them BEFORE scoring...a synthetic placeholder is inserted so the scorer does not crash and is removed afterwards so the output file stays faithful to the source dataset.
            abstain_idx = find_cat5_abstention_rows(out_data["qa"])
            for i in abstain_idx:
                out_data["qa"][i]["answer"] = "MISSING_GROUND_TRUTH"

            exact_matches, lengths, recall = eval_question_answering(
                out_data["qa"], prediction_key
            )

            for i in range(len(out_data["qa"])):
                if i in abstain_idx:
                    # NOTE to myself:
                    # Abstention accuracy: 1 iff (if and only if) the model rejected the
                    # adversarial distractor in favour of "Not mentioned in
                    # the conversation". Binary, so accuracy == F1.
                    pred = (
                        str(out_data["qa"][i].get(prediction_key, ""))
                        .strip()
                        .rstrip(".")
                        .lower()
                    )
                    exact_matches[i] = 1.0 if pred == NOT_MENTIONED else 0.0
                out_data["qa"][i][model_key + "_f1"] = round(exact_matches[i], 3)
                if args.use_rag and len(recall) > 0:
                    out_data["qa"][i][model_key + "_recall"] = round(recall[i], 3)

            # remove the synthetic placeholder so the persisted output does not mess and masquerade as dataset ground truth.
            for i in abstain_idx:
                out_data["qa"][i].pop("answer", None)

            # re-flush now that scores are attached
            atomic_write(args.out_file, out_samples)
            logger.info("Sample %s evaluated and re-flushed with scores.", sid)
        except Exception as exc:
            logger.error(
                "Evaluation FAILED for sample %s: %s. "
                "Raw predictions are already saved; continuing to next sample.",
                sid,
                exc,
                exc_info=True,
            )

    # aggregate stats final
    stats_path = args.out_file.replace(".json", "_stats.json")
    analyze_aggr_acc(
        args.data_file,
        args.out_file,
        stats_path,
        model_key,
        model_key + "_f1",
        rag=args.use_rag,
    )
    logger.info("Done. Results: %s | Stats: %s", args.out_file, stats_path)


# runnnnn lets goooo!
if __name__ == "__main__":
    main()
