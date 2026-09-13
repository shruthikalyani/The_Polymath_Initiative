class ArchiveMemory:
    def __init__(self, max_items=200):
        self.max_items = max_items
        self.memory_vault = []

    def add(self, memory):
        # Store old memories here when they're no longer needed up front
        self.memory_vault.append(memory)

        # The vault has a limit too. not like gojo 0o
        if len(self.memory_vault) > self.max_items:
            forgotten_memory = self.memory_vault.pop(0)
            return forgotten_memory

        return None

    def get_all(self):
        # Bring out everything currently hiding .
        return self.memory_vault.copy()

    def clear(self):
        # Empty  but keep the old memories in case we need them
        old_memories = self.memory_vault.copy()
        self.memory_vault = []
        return old_memories

    def __len__(self):
        return len(self.memory_vault)