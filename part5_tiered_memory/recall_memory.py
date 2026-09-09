class RecallMemory:
    def __init__(self, max_items=50):
        self.max_items = max_items
        self.memory_bank = []

    def add(self, memory, memory_type="raw"):
        # Put the new memory into the little memory bank
        memory_card = {
            "type": memory_type,
            "content": memory
        }

        self.memory_bank.append(memory_card)

        # If the bank gets too crowded, yeet the oldest memory 🙃🙃🙃
        if len(self.memory_bank) > self.max_items:
            forgotten_memory = self.memory_bank.pop(0)
            return forgotten_memory

        return None

    def get_all(self):
        # Give back everything we've remembered.
        return self.memory_bank.copy()

    def get_raw(self):
        # Grab the memories that are still in their raw form.
        return [
            memory for memory in self.memory_bank
            if memory["type"] == "raw"
        ]

    def get_summaries(self):
        # Grab the memories that have been compressed into summaries
        return [
            memory for memory in self.memory_bank
            if memory["type"] == "summary"
        ]

    def clear(self):
        # Empty the memory bank, but don't lose the receipts
        old_memories = self.memory_bank.copy()
        self.memory_bank = []
        return old_memories

    def __len__(self):
        return len(self.memory_bank)