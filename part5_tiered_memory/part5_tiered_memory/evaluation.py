import re
import string
from datetime import datetime


def normalize_date_format(answer):
    """
    Convert common date formats into one standard format.

    Examples:
        May 7, 2023
        7 May 2023
        7 May, 2023

    All become:

        2023-05-07
    """

    if not answer:
        return answer

    # Remove a final full stop.
    cleaned = str(answer).strip().rstrip(".")

    # Try the original text and a title-cased version.
    # Title casing lets us handle answers such as
    # "may 7, 2023" as well as "May 7, 2023".
    candidates = [
        cleaned,
        cleaned.title(),
    ]

    date_patterns = [
        "%B %d, %Y",
        "%B %d %Y",
        "%d %B, %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%d %b, %Y",
        "%d %b %Y",
    ]

    for candidate in candidates:
        for pattern in date_patterns:
            try:
                parsed_date = datetime.strptime(
                    candidate,
                    pattern
                )

                return parsed_date.strftime(
                    "%Y-%m-%d"
                )

            except ValueError:
                continue

    return cleaned


def normalize_answer(answer):
    """
    Normalize an answer before comparison.

    Handles:
    - capitalization
    - punctuation
    - whitespace
    - common date formatting differences
    """

    if answer is None:
        return ""

    answer = str(answer).strip()

    # Check for a date BEFORE removing punctuation or
    # changing the format of the answer.
    normalized_date = normalize_date_format(
        answer
    )

    # If the answer was successfully recognized as a date,
    # use the standard YYYY-MM-DD representation.
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}",
        normalized_date
    ):
        return normalized_date

    # Otherwise use normal text normalization.
    answer = answer.lower()

    # Remove punctuation.
    answer = answer.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Normalize whitespace.
    answer = " ".join(
        answer.split()
    )

    return answer


def exact_match(prediction, ground_truth):
    """Return 1 when normalized answers are identical."""

    prediction = normalize_answer(
        prediction
    )

    ground_truth = normalize_answer(
        ground_truth
    )

    return int(
        prediction == ground_truth
    )


def token_f1(prediction, ground_truth):
    """Calculate token-level F1."""

    prediction_tokens = normalize_answer(
        prediction
    ).split()

    ground_truth_tokens = normalize_answer(
        ground_truth
    ).split()

    if not prediction_tokens or not ground_truth_tokens:
        return float(
            prediction_tokens == ground_truth_tokens
        )

    prediction_counts = {}

    for token in prediction_tokens:
        prediction_counts[token] = (
            prediction_counts.get(token, 0) + 1
        )

    ground_truth_counts = {}

    for token in ground_truth_tokens:
        ground_truth_counts[token] = (
            ground_truth_counts.get(token, 0) + 1
        )

    common_tokens = 0

    for token in prediction_counts:
        if token in ground_truth_counts:
            common_tokens += min(
                prediction_counts[token],
                ground_truth_counts[token]
            )

    if common_tokens == 0:
        return 0.0

    precision = (
        common_tokens
        / len(prediction_tokens)
    )

    recall = (
        common_tokens
        / len(ground_truth_tokens)
    )

    return (
        2 * precision * recall
        / (precision + recall)
    )


def score_answer(prediction, ground_truth):
    """Return the primary evaluation metrics."""

    return {
        "exact_match": exact_match(
            prediction,
            ground_truth
        ),
        "f1": token_f1(
            prediction,
            ground_truth
        ),
    }


def main():
    """Sanity check for the evaluation system."""

    print(
        "===== PART 5 EVALUATION TEST ====="
    )

    test_cases = [
        (
            "May 7, 2023.",
            "7 May 2023"
        ),
        (
            "7 May, 2023",
            "7 May 2023"
        ),
        (
            "may 7 2023",
            "7 May 2023"
        ),
        (
            "Hello World",
            "hello world"
        ),
    ]

    for prediction, ground_truth in test_cases:

        scores = score_answer(
            prediction,
            ground_truth
        )

        print("\nPrediction:", prediction)
        print("Ground truth:", ground_truth)

        print(
            "Normalized prediction:",
            normalize_answer(prediction)
        )

        print(
            "Normalized ground truth:",
            normalize_answer(ground_truth)
        )

        print(
            "Exact Match:",
            scores["exact_match"]
        )

        print(
            "F1:",
            scores["f1"]
        )

    print(
        "\n===== EVALUATION TEST COMPLETE ====="
    )


if __name__ == "__main__":
    main()