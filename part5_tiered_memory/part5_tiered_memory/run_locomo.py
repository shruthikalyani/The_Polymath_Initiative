import json

from .config import DATASET_PATH


def load_dataset():
    """Load the original LoCoMo dataset without changing it."""

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset


def get_sessions(conversation):
    """
    Collect the numbered conversation sessions in order.

    We only want session_1, session_2, etc.
    The date/time fields are useful metadata but are not conversation turns.
    """

    sessions = []

    for key, value in conversation.items():
        if key.startswith("session_") and key.split("_")[-1].isdigit():
            sessions.append((int(key.split("_")[-1]), value))

    # Keep the conversation chronological.
    sessions.sort(key=lambda item: item[0])

    return sessions


def main():
    dataset = load_dataset()

    print("===== PART 5 LOCOMO LOADER =====")
    print(f"Samples loaded: {len(dataset)}")

    first_sample = dataset[0]
    sessions = get_sessions(first_sample["conversation"])

    print(f"Sessions in first sample: {len(sessions)}")

    first_session_number, first_session = sessions[0]

    print(f"First session: session_{first_session_number}")
    print(f"Turns in first session: {len(first_session)}")

    print("\nFIRST TURN")
    print(first_session[0])

    print("\nLAST TURN")
    print(first_session[-1])

    print("\nQA COUNT")
    print(len(first_sample["qa"]))

    print("\nFIRST QA")
    print(first_sample["qa"][0])


if __name__ == "__main__":
    main()