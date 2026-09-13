import json

from .config import DATASET_PATH
from .memory_manager import MemoryManager


def load_first_sample():
    """Load one LoCoMo sample for a small memory test."""

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset[0]


def get_sessions(conversation):
    """Return the actual conversation sessions in chronological order."""

    sessions = []

    for key, value in conversation.items():
        if key.startswith("session_") and key.split("_")[-1].isdigit():
            sessions.append((int(key.split("_")[-1]), value))

    sessions.sort(key=lambda item: item[0])

    return sessions


def main():
    sample = load_first_sample()

    memory = MemoryManager()

    sessions = get_sessions(sample["conversation"])

    print("===== PART 5 LOCOMO MEMORY TEST =====")

    for session_number, turns in sessions:
        print(f"\nLoading session {session_number}...")

        for turn in turns:
            memory.add_turn(
                turn["speaker"],
                turn["text"]
            )

    print("\n===== FINAL MEMORY STATE =====")

    print("Working memory:", len(memory.get_working_memory()))
    print("Recall memory:", len(memory.get_recall_memory()))
    print("Archive memory:", len(memory.get_archive_memory()))

    print("\n===== SAMPLE WORKING MEMORY =====")
    print(memory.get_working_memory()[:2])

    print("\n===== SAMPLE RECALL MEMORY =====")
    print(memory.get_recall_memory()[:2])

    print("\n===== SAMPLE ARCHIVE MEMORY =====")
    print(memory.get_archive_memory()[:2])


if __name__ == "__main__":
    main()