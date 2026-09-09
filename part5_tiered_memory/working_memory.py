class WorkingMemory:
    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.turns = []

    def add(self, speaker, message):
        # Keep the newest conversation in the "brain".
        new_turn = {
            "speaker": speaker,
            "message": message
        }

        self.turns.append(new_turn)

        # Memory is full, so the oldest thing gets kicked out.
        if len(self.turns) > self.max_turns:
            old_turn = self.turns.pop(0)
            return old_turn

        return None

    def get_turns(self):
        # Give us a copy so the original memory stays safe.
        return self.turns.copy()

    def clear(self):
        # Empty the brain, but save what was there first.
        old_turns = self.turns.copy()
        self.turns = []
        return old_turns

    def __len__(self):
        return len(self.turns)