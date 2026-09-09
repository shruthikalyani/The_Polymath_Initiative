class MemorySummarizer:
    def __init__(self, model=None):
        # The model gets plugged in later.
        self.model = model

    def make_summary_prompt(self, turns):
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
- events
- details that could help answer future questions

Do not make up information.

Conversation:
{conversation}

Memory summary:
""".strip()

    def summarize(self, turns):
        if not turns:
            return ""

        prompt = self.make_summary_prompt(turns)

        # The actual Llama call will be connected later.
        if self.model is None:
            return prompt

        return self.model.generate(prompt)