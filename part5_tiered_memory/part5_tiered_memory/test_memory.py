from part5_tiered_memory.memory_manager import MemoryManager


memory = MemoryManager()


# Add enough turns to overflow working memory.
for i in range(15):
    memory.add_turn(
        "User",
        f"Test conversation number {i}. I am studying robotics."
    )


print("\n===== MEMORY TEST =====")

print("\nWorking memory:")
print(len(memory.get_working_memory()))
print(memory.get_working_memory())

print("\nRecall memory:")
print(len(memory.get_recall_memory()))
print(memory.get_recall_memory())

print("\nArchive memory:")
print(len(memory.get_archive_memory()))
print(memory.get_archive_memory())

print("\nSearching for robotics:")
print(memory.retrieve("robotics"))
print("\n===== RETRIEVAL DEBUG =====")

recall = memory.get_recall_memory()

print("Number of recall memories:", len(recall))

for item in recall:
    print("CONTENT:", item["content"])
    print("MESSAGE:", item["content"]["message"])
    print("HAS ROBOTICS:", "robotics" in item["content"]["message"].lower())

print("\nDIRECT RETRIEVAL:")
print(memory.retrieve("robotics"))