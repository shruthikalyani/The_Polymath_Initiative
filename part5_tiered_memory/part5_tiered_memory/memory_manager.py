from .working_memory import WorkingMemory
from .recall_memory import RecallMemory
from .archive_memory import ArchiveMemory
from .summarizer import MemorySummarizer
from .config import (
    WORKING_MEMORY_TURNS,
    RECALL_MEMORY_LIMIT,
    ARCHIVE_MEMORY_LIMIT,
    SUMMARY_TRIGGER_TURNS,
)


class MemoryManager:
    def __init__(self, model=None):
        # Three levels of memory:
        # working = recent stuff
        # recall = useful stuff
        # archive = older stuff
        self.working_memory = WorkingMemory(
            WORKING_MEMORY_TURNS
        )

        self.recall_memory = RecallMemory(
            RECALL_MEMORY_LIMIT
        )

        self.archive_memory = ArchiveMemory(
            ARCHIVE_MEMORY_LIMIT
        )

        self.summarizer = MemorySummarizer(model)

        # Turns waiting to be compressed.
        self.pending_summary = []

    def add_turn(
        self,
        speaker,
        message,
        metadata=None
    ):
        """Add one conversation turn to the memory system."""

        kicked_out = self.working_memory.add(
            speaker,
            message,
            metadata
        )

        # When working memory is full,
        # the oldest turn gets paged out.
        if kicked_out is not None:
            self._move_to_recall(
                kicked_out
            )

    def _move_to_recall(self, turn):
        """Move an old working-memory turn into recall memory."""

        raw_memory = {
            "speaker": turn["speaker"],
            "message": turn["message"],
            "metadata": turn.get(
                "metadata",
                {}
            )
        }

        # Keep the original turn as raw memory.
        forgotten_memory = self.recall_memory.add(
            raw_memory,
            memory_type="raw",
        )

        # If recall is full, page the oldest
        # memory into the archive.
        if forgotten_memory is not None:
            self.archive_memory.add(
                forgotten_memory
            )

        # Add this turn to the next summary batch.
        self.pending_summary.append(
            turn
        )

        # Once enough turns pile up,
        # Llama compresses them.
        if (
            len(self.pending_summary)
            >= SUMMARY_TRIGGER_TURNS
        ):
            self._make_summary()

    def _make_summary(self):
        """Use Llama to compress old turns into a summary."""

        if not self.pending_summary:
            return None

        summary = self.summarizer.summarize(
            self.pending_summary
        )

        if summary:
            self.recall_memory.add(
                summary,
                memory_type="summary",
            )

        self.pending_summary = []

        return summary

    def _memory_text(self, memory):
        """
        Turn any memory object into readable text.

        Session dates are kept visible so Llama can reason
        about relative dates such as "yesterday".
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

    def _get_keyword_candidates(
        self,
        query,
        limit=20
    ):
        """
        First retrieval stage.

        Find a manageable set of candidate memories
        using word overlap.

        This does NOT make the final retrieval decision.
        """

        import re

        stop_words = {
            "what", "when", "where", "who",
            "why", "how", "did", "does",
            "do", "is", "are", "was",
            "were", "the", "a", "an",
            "and", "or", "to", "of",
            "about", "say", "said",
            "tell", "me", "for",
            "in", "on", "with",
        }

        query_words = {
            word
            for word in re.findall(
                r"\b\w+\b",
                query.lower()
            )
            if word not in stop_words
        }

        all_memories = (
            self.working_memory.get_turns()
            + self.recall_memory.get_all()
            + self.archive_memory.get_all()
        )

        scored_memories = []

        for memory in all_memories:
            text = self._memory_text(
                memory
            )

            memory_words = set(
                re.findall(
                    r"\b\w+\b",
                    text.lower()
                )
            )

            score = len(
                query_words & memory_words
            )

            if score > 0:
                scored_memories.append(
                    (score, memory)
                )

        scored_memories.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return [
            memory
            for score, memory
            in scored_memories[:limit]
        ]

    def _llama_retrieval_judgment(
        self,
        query,
        candidates,
        limit=5
    ):
        """
        Second retrieval stage.

        Llama decides which candidate memories
        are actually relevant to the question.

        LoCoMo evidence IDs are NOT used for retrieval.
        """

        model = self.summarizer.model

        if model is None:
            return candidates[:limit]

        if not candidates:
            return []

        candidate_text = []

        for index, memory in enumerate(
            candidates,
            start=1
        ):
            candidate_text.append(
                f"[{index}] "
                f"{self._memory_text(memory)}"
            )

        joined_candidates = "\n".join(
            candidate_text
        )

        prompt = f"""
You are the retrieval component of a conversational
memory system.

Your job is to choose the memories that contain
information needed to answer the question.

Question:
{query}

Candidate memories:
--------------------
{joined_candidates}
--------------------

Choose up to {limit} relevant memories.

Pay close attention to:
- names
- events
- dates
- session dates
- relative dates such as yesterday or last week
- relationships
- specific activities

Only choose memories that are useful for answering
the question.

Do not use outside knowledge.
Do not invent information.
Do not explain your choice.

Return ONLY the candidate numbers separated by commas.

Example:
2,5,7

Selected memories:
""".strip()

        response = model.generate(
            prompt,
            max_new_tokens=32
        )

        selected_numbers = []

        for number in response.split(","):
            number = number.strip()

            if number.isdigit():
                index = int(number)

                if (
                    1 <= index <= len(candidates)
                    and index not in selected_numbers
                ):
                    selected_numbers.append(
                        index
                    )

        selected_memories = [
            candidates[index - 1]
            for index in selected_numbers[:limit]
        ]

        # If Llama gives an unusable response,
        # keep the strongest candidates instead.
        if not selected_memories:
            return candidates[:limit]

        return selected_memories

    def retrieve(
        self,
        query,
        limit=5
    ):
        """
        Retrieve memories relevant to a question.

        Stage 1:
        Python finds candidate memories.

        Stage 2:
        Llama makes the actual retrieval judgment.
        """

        candidates = self._get_keyword_candidates(
            query,
            limit=20
        )

        return self._llama_retrieval_judgment(
            query,
            candidates,
            limit=limit
        )

    def move_old_memories_to_archive(self):
        """Page overflowing recall memories into the archive."""

        while (
            len(self.recall_memory)
            > RECALL_MEMORY_LIMIT
        ):
            memories = self.recall_memory.get_all()

            if not memories:
                break

            oldest_memory = memories[0]

            self.recall_memory.memory_bank.pop(
                0
            )

            self.archive_memory.add(
                oldest_memory
            )

    def get_working_memory(self):
        return self.working_memory.get_turns()

    def get_recall_memory(self):
        return self.recall_memory.get_all()

    def get_archive_memory(self):
        return self.archive_memory.get_all()

    def get_memory_state(self):
        return {
            "working": self.get_working_memory(),
            "recall": self.get_recall_memory(),
            "archive": self.get_archive_memory(),
        }

    def clear(self):
        self.working_memory.clear()
        self.recall_memory.clear()
        self.archive_memory.clear()
        self.pending_summary = []