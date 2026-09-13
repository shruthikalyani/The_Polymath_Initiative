from .config import MODEL_ID
from .locomo_adapter import ( load_dataset, get_all_turns , get_all_summaries ,)
from .memory_manager import MemoryManager
from .model import LlamaModel
from .qa_engine import answer_question
from .evaluation import score_answer
from .results_writer import save_results


def add_turn_with_date(memory, turn):
    """
    Add a LoCoMo turn while keeping its session date visible.

    The date stays attached to the memory because some
    questions depend on relative dates like "yesterday".
    """

    date = turn.get("session_date_time")

    if date:
        message = (
            f"[Session date: {date}] "
            f"{turn['text']}"
        )
    else:
        message = turn["text"]

    memory.add_turn(
        turn["speaker"],
        message,
        metadata={
            "dia_id": turn.get("dia_id"),
            "session": turn.get("session"),
            "session_date_time": turn.get(
                "session_date_time"
            ),
        },
    )


def run_one_sample(sample, model, sample_index):
    """Replay one complete LoCoMo conversation and test its QA."""

    memory = MemoryManager(model)

    turns = get_all_turns(sample)

    print(
        f"\n===== SAMPLE {sample_index} ====="
    )

    print(
        f"Conversation turns: {len(turns)}"
    )

    # Replay the complete conversation through the
    # tiered memory system.
    for turn in turns:
        add_turn_with_date(
            memory,
            turn
        )
        # LoCoMo already provides session summaries.
    # Use those summaries instead of asking Llama
    # to generate new ones.
    summaries = get_all_summaries(sample)

    print(
        f"Dataset summaries loaded: {len(summaries)}"
    )

    for summary in summaries:

        summary_text = (
            f"[Session date: "
            f"{summary['session_date_time']}] "
            f"{summary['summary']}"
        )

        memory.recall_memory.add(
            summary_text,
            memory_type="summary"
        )

    questions = sample.get("qa", [])

    print(
        f"Questions available: {len(questions)}"
    )

    # IMPORTANT:
    # No [:10] here.
    # We want every available question.
    results = []

    for index, qa in enumerate(
        questions,
        start=1
    ):

        question = qa["question"]

        # Some LoCoMo adversarial questions may not
        # contain a normal ground-truth answer.
        ground_truth = qa.get("answer")

        prediction, retrieved = answer_question(
            memory,
            model,
            question
        )

        result = {
            "sample_index": sample_index,
            "question_index": index,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "category": qa.get("category"),
            "evidence": qa.get(
                "evidence",
                []
            ),
            "retrieved_memories": retrieved,
        }

        # Only calculate EM/F1 when the dataset provides
        # a normal ground-truth answer.
        if ground_truth is not None:

            scores = score_answer(
                prediction,
                ground_truth
            )

            result["exact_match"] = (
                scores["exact_match"]
            )

            result["f1"] = scores["f1"]

        else:
            result["exact_match"] = None
            result["f1"] = None

        results.append(result)

        print(
            f"[{index}/{len(questions)}] "
            f"{question}"
        )

        print(
            f"Prediction: {prediction}"
        )

        print(
            f"Exact Match: "
            f"{result['exact_match']}"
        )

        print(
            f"F1: "
            f"{result['f1']}"
        )

        print()

    return results


def main():
    """Run the complete Part 5 LoCoMo experiment."""

    print(
        "===== PART 5 LOCOMO QA EXPERIMENT ====="
    )

    dataset = load_dataset()

    print(
        f"LoCoMo samples available: "
        f"{len(dataset)}"
    )

    model = LlamaModel(MODEL_ID)

    all_results = []

    # Run EVERY sample in the dataset.
    for sample_index, sample in enumerate(
        dataset
    ):

        sample_results = run_one_sample(
            sample,
            model,
            sample_index
        )

        all_results.extend(
            sample_results
        )

        # Save progress after every sample.
        # If the API stops halfway through, we still
        # keep everything that has already finished.
        experiment_results = {
            "experiment": (
                "part5_summaries_raw_tiered_paged"
            ),
            "model": MODEL_ID,
            "dataset": "LoCoMo",
            "status": "running",
            "samples_tested": sample_index + 1,
            "total_samples": len(dataset),
            "questions_processed": len(
                all_results
            ),
            "results": all_results,
        }

        save_results(
            experiment_results,
            filename="locomo_results.json"
        )

        print(
            f"Saved progress after sample "
            f"{sample_index}."
        )

    # Final result package.
    experiment_results = {
        "experiment": (
            "part5_summaries_raw_tiered_paged"
        ),
        "model": MODEL_ID,
        "dataset": "LoCoMo",
        "status": "complete",
        "samples_tested": len(dataset),
        "total_samples": len(dataset),
        "questions_processed": len(
            all_results
        ),
        "results": all_results,
    }

    output_path = save_results(
        experiment_results,
        filename="locomo_results.json"
    )

    print(
        "\n======================================"
    )

    print(
        "===== PART 5 EXPERIMENT COMPLETE ====="
    )

    print(
        f"Samples tested: {len(dataset)}"
    )

    print(
        f"Questions processed: "
        f"{len(all_results)}"
    )

    print(
        "\nRESULTS JSON:"
    )

    print(
        output_path
    )

    print(
        "\n======================================"
    )


if __name__ == "__main__":
    main()