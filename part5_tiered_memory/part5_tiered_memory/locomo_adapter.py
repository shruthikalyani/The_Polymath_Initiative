import json

from .config import DATASET_PATH


def load_dataset():
    """Load the original LoCoMo dataset."""

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def get_sessions(conversation):
    """
    Get the actual conversation sessions.

    Date/time fields are ignored here because they are
    metadata rather than conversation turns.
    """

    sessions = []

    for key, value in conversation.items():

        if (
            key.startswith("session_")
            and key.split("_")[-1].isdigit()
        ):
            session_number = int(
                key.split("_")[-1]
            )

            if isinstance(value, list):
                sessions.append(
                    (session_number, value)
                )

    sessions.sort(
        key=lambda item: item[0]
    )

    return sessions


def get_session_date(
    conversation,
    session_number
):
    """Get the date/time attached to a session."""

    key = f"session_{session_number}_date_time"

    return conversation.get(key)


def get_session_summary(
    sample,
    session_number
):
    """
    Get the summary already provided by LoCoMo.

    These summaries come directly from the dataset.
    Llama does not generate these summaries.
    """

    session_summaries = sample.get(
        "session_summary",
        {}
    )

    key = (
        f"session_{session_number}_summary"
    )

    return session_summaries.get(key)


def get_all_turns(sample):
    """
    Flatten all conversation sessions into one
    chronological list of turns.
    """

    conversation = sample["conversation"]

    sessions = get_sessions(
        conversation
    )

    all_turns = []

    for session_number, turns in sessions:

        session_date = get_session_date(
            conversation,
            session_number
        )

        for turn in turns:

            all_turns.append({
                "speaker": turn["speaker"],
                "text": turn["text"],
                "dia_id": turn["dia_id"],
                "session": session_number,
                "session_date_time": session_date,
            })

    return all_turns


def get_all_summaries(sample):
    """
    Get all session-level summaries supplied
    by the LoCoMo dataset.
    """

    conversation = sample["conversation"]

    sessions = get_sessions(
        conversation
    )

    summaries = []

    for session_number, _ in sessions:

        summary = get_session_summary(
            sample,
            session_number
        )

        if summary:

            summaries.append({
                "session": session_number,
                "session_date_time": (
                    get_session_date(
                        conversation,
                        session_number
                    )
                ),
                "summary": summary,
            })

    return summaries


def get_evidence_map(sample):
    """
    Create a quick lookup from dialogue ID
    to the original conversation turn.
    """

    evidence_map = {}

    for turn in get_all_turns(sample):

        evidence_map[
            turn["dia_id"]
        ] = turn

    return evidence_map


def main():
    """Sanity check for the LoCoMo adapter."""

    dataset = load_dataset()

    sample = dataset[0]

    turns = get_all_turns(
        sample
    )

    summaries = get_all_summaries(
        sample
    )

    evidence_map = get_evidence_map(
        sample
    )

    print(
        "===== LOCOMO ADAPTER TEST ====="
    )

    print(
        "Total turns:",
        len(turns)
    )

    print(
        "Total session summaries:",
        len(summaries)
    )

    print(
        "\nFIRST SUMMARY"
    )

    if summaries:

        print(
            summaries[0]
        )

    print(
        "\nLAST SUMMARY"
    )

    if summaries:

        print(
            summaries[-1]
        )

    print(
        "\nD1:3 EVIDENCE"
    )

    print(
        evidence_map["D1:3"]
    )


if __name__ == "__main__":
    main()