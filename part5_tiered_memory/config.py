from pathlib import Path
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "locomo10.json"
RESULTS_DIR = PROJECT_ROOT / "results_part5_tiered_memory"
LOGS_DIR = PROJECT_ROOT / "logs_part5_tiered_memory"
# Model
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# Memory settings
WORKING_MEMORY_TURNS = 10
RECALL_MEMORY_LIMIT = 50
ARCHIVE_MEMORY_LIMIT = 200
# Summarize after this many new turns
SUMMARY_TRIGGER_TURNS = 10
# Experiment settings
EXPERIMENT_NAME = "part5_summaries_raw_tiered_paged"
RANDOM_SEED = 42
# Create folders for the experiment outputs
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
