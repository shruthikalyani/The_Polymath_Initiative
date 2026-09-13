from part5_tiered_memory.config import MODEL_ID
from part5_tiered_memory.memory_manager import MemoryManager
from part5_tiered_memory.model import LlamaModel


# Connect the memory system to Llama.
model = LlamaModel(MODEL_ID)
memory = MemoryManager(model)


# We use lots of turns here so the memory tiers actually get tested.
# 70 turns is enough to overflow recall memory and reach the archive.
for i in range(70):
    memory.add_turn(
        "User",
        f"Conversation turn {i}: I am working on robotics project number {i}."
    )


print("\n===== PART 5 PAGING STRESS TEST =====")

print("\nWORKING MEMORY")
print("Count:", len(memory.get_working_memory()))

print("\nRECALL MEMORY")
print("Count:", len(memory.get_recall_memory()))

print("\nARCHIVE MEMORY")
print("Count:", len(memory.get_archive_memory()))

print("\nMEMORY TIER CHECK")

if len(memory.get_working_memory()) == 10:
    print("Working memory: PASS")
else:
    print("Working memory: CHECK")

if len(memory.get_recall_memory()) <= 50:
    print("Recall memory limit: PASS")
else:
    print("Recall memory limit: CHECK")

if len(memory.get_archive_memory()) > 0:
    print("Archive paging: PASS")
else:
    print("Archive paging: CHECK")

print("\nARCHIVED MEMORY SAMPLE")

archive = memory.get_archive_memory()

for item in archive[:3]:
    print(item)

print("\nARCHIVE RETRIEVAL TEST")

results = memory.retrieve("robotics project number 0")

print("Matches:", results)

print("\n===== PAGING TEST COMPLETE =====")