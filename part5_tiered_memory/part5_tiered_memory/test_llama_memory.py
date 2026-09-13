from part5_tiered_memory.config import MODEL_ID
from part5_tiered_memory.memory_manager import MemoryManager
from part5_tiered_memory.model import LlamaModel


# Connect our memory manager to Llama.
model = LlamaModel(MODEL_ID)
memory = MemoryManager(model)


# Add 20 turns so the memory system has to start paging things around.
conversation = [
    ("User", "My name is Alex."),
    ("Assistant", "Nice to meet you, Alex."),
    ("User", "I am studying robotics."),
    ("Assistant", "Robotics is a great field."),
    ("User", "I am especially interested in underwater robots."),
    ("Assistant", "That sounds like an interesting project."),
    ("User", "I want to build one in the future."),
    ("Assistant", "You could explore underwater navigation."),
    ("User", "I also enjoy working with Arduino."),
    ("Assistant", "Arduino can be useful for prototypes."),
    ("User", "My favourite project so far is a small robot."),
    ("Assistant", "That gives you useful hands-on experience."),
    ("User", "I want to learn more about sensors."),
    ("Assistant", "Sensors are important in robotics."),
    ("User", "I am also interested in artificial intelligence."),
    ("Assistant", "AI and robotics work well together."),
    ("User", "My long-term goal is underwater robotics."),
    ("Assistant", "That connects nicely with your interests."),
    ("User", "I want to combine robotics and AI."),
    ("Assistant", "That could become a strong project direction."),
]


for speaker, message in conversation:
    memory.add_turn(speaker, message)


print("\n===== PART 5 LLAMA MEMORY TEST =====")

print("\nWORKING MEMORY")
print("Count:", len(memory.get_working_memory()))
print(memory.get_working_memory())

print("\nRECALL MEMORY")
print("Count:", len(memory.get_recall_memory()))

for item in memory.get_recall_memory():
    print("\nTYPE:", item["type"])
    print("CONTENT:", item["content"])

print("\nARCHIVE MEMORY")
print("Count:", len(memory.get_archive_memory()))

print("\nPENDING SUMMARY")
print("Count:", len(memory.pending_summary))

print("\nRETRIEVAL TEST")
print(memory.retrieve("robotics"))

print("\n===== TEST COMPLETE =====")