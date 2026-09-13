import json
from pathlib import Path

from .config import RESULTS_DIR


def save_results(results, filename="locomo_results.json"):
    """
    Save experiment results as a JSON file.

    Keeping this separate from the experiment runner makes it
    easier to inspect or reuse the results later.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = RESULTS_DIR / filename

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    return output_path


def load_results(filename="locomo_results.json"):
    """Load previously saved experiment results."""

    output_path = RESULTS_DIR / filename

    with open(
        output_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def main():
    """Small test for the JSON writer."""

    print("===== PART 5 RESULTS WRITER TEST =====")

    test_results = {
        "experiment": "part5_summaries_raw_tiered_paged",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "status": "test",
        "results": [
            {
                "question": "Test question",
                "ground_truth": "Test answer",
                "prediction": "Test answer",
                "exact_match": 1,
                "f1": 1.0
            }
        ]
    }

    output_path = save_results(
        test_results,
        filename="test_results.json"
    )

    print("Results saved to:")
    print(output_path)

    print("===== RESULTS WRITER TEST COMPLETE =====")


if __name__ == "__main__":
    main()