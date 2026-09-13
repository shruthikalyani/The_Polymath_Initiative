class MemorySummarizer:
    def __init__(self, model=None):
        # Llama gets plugged in here.
        self.model = model

    def make_summary_prompt(self, turns):
        """Build the little memory-compression prompt."""

        if not turns:
            return ""

        conversation = "\n".join(
            f"{turn['speaker']}: {turn['message']}"
            for turn in turns
        )

        return f"""
Summarize the following conversation for long-term memory.

Keep:
- important facts
- names
- dates
- relationships
- preferences
- important events
- details that may help answer future questions

Keep the summary factual and concise.

Do not invent or guess information.

Conversation:
{conversation}

Memory summary:
""".strip()

    def summarize(self, turns):
        """Turn a batch of old turns into one useful memory."""

        if not turns:
            return ""

        prompt = self.make_summary_prompt(turns)

        # If no model is connected, return the prompt for debugging.
        if self.model is None:
            return prompt

        # Llama does the actual memory compression.
        return self.model.generate(
            prompt,
            max_new_tokens=256
        )