from .config import MODEL_ID
from .memory_manager import MemoryManager
from .model import LlamaModel


def format_memory(memory):
    """
    Turn any memory type into readable text for Llama.

    Session dates are included because they are important
    for questions involving relative dates.
    """

    if (
        "speaker" in memory
        and "message" in memory
    ):
        metadata = memory.get(
            "metadata",
            {}
        )

        date = metadata.get(
            "session_date_time"
        )

        if date:
            return (
                f"[Session date: {date}] "
                f"{memory.get('speaker', 'Unknown')}: "
                f"{memory.get('message', '')}"
            )

        return (
            f"{memory.get('speaker', 'Unknown')}: "
            f"{memory.get('message', '')}"
        )

    content = memory.get(
        "content",
        ""
    )

    if isinstance(content, dict):
        metadata = content.get(
            "metadata",
            {}
        )

        date = metadata.get(
            "session_date_time"
        )

        if date:
            return (
                f"[Session date: {date}] "
                f"{content.get('speaker', 'Unknown')}: "
                f"{content.get('message', '')}"
            )

        return (
            f"{content.get('speaker', 'Unknown')}: "
            f"{content.get('message', '')}"
        )

    return str(content)


def build_answer_prompt(
    question,
    memories
):
    """Build the final question-answering prompt."""

    memory_text = [
        format_memory(memory)
        for memory in memories
    ]

    joined_memories = "\n".join(
        memory_text
    )

    return f"""
You are answering a question about a previous conversation.

Use ONLY the conversation memory provided below.

Do not say that the conversation memory is missing
if relevant memory is provided.

Do not invent facts.

If the answer is directly stated in the memory,
answer directly.

If the question requires simple reasoning, use only
information contained in the memory.

For questions involving relative dates such as
"yesterday", "tomorrow", "last week", or
"two days ago", use the session date provided
in the memory to calculate the actual date.

For example, if a conversation says:
"[Session date: 8 May, 2023] Caroline: I went there yesterday."

then "yesterday" means 7 May 2023.

Do not confuse the conversation session date
with the date of the event being described.

Conversation memory:
--------------------
{joined_memories}
--------------------

Question:
{question}

Give ONLY the answer to the question.

Do not explain your reasoning.
Do not repeat the question.
Do not add introductory phrases.
Do not add extra context.

For date questions, give the specific date if
the memory allows it.

For name questions, give only the name.

For number questions, give only the number.

Answer:
""".strip()


def answer_question(
    memory,
    model,
    question
):
    """Retrieve memories and ask Llama for the answer."""

    memories = memory.retrieve(
        question,
        limit=10
    )

    prompt = build_answer_prompt(
        question,
        memories
    )

    answer = model.generate(
        prompt,
        max_new_tokens=128
    )

    return answer, memories


def main():
    """Small test of the Part 5 QA engine."""

    print(
        "===== PART 5 QA ENGINE TEST ====="
    )

    model = LlamaModel(MODEL_ID)

    memory = MemoryManager(
        model
    )

    conversation = [
        (
            "Caroline",
            "I went to a LGBTQ support group yesterday."
        ),
        (
            "Melanie",
            "That sounds like it was meaningful."
        ),
        (
            "Caroline",
            "It really helped me feel accepted."
        ),
        (
            "Melanie",
            "I'm glad you found that support."
        ),
    ]

    for speaker, message in conversation:
        memory.add_turn(
            speaker,
            message
        )

    question = (
        "What did Caroline say about the support group?"
    )

    answer, memories = answer_question(
        memory,
        model,
        question
    )

    print("\nQUESTION")
    print(question)

    print("\nRETRIEVED MEMORIES")

    for memory_item in memories:
        print(memory_item)

    print("\nLLAMA ANSWER")
    print(answer)

    print(
        "\n===== QA ENGINE TEST COMPLETE ====="
    )


if __name__ == "__main__":
    main()