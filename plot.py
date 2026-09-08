import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------
# Your actual data from the JSON file (the "new data")
# --------------------------------------------------------------------

data = {
    "meta-llama/Llama-3.1-8B-Instruct_full_dialog_top_5": {
        "category_counts": {
            "2": 321,
            "3": 96,
            "1": 282,
            "4": 841,
            "5": 446,
        },
        "cum_accuracy_by_category": {
            "2": 55.75600000000005,
            "3": 10.693000000000001,
            "1": 88.11100000000006,
            "4": 470.0250000000001,
            "5": 177.0,
        },
        "recall_by_category": {
            "2": 1.0,
            "3": 0.9583333333333334,
            "1": 1.0,
            "4": 1.0,
            "5": 1.0,
        },
    }
}

# Extract counts and cumulative accuracies
counts = data["meta-llama/Llama-3.1-8B-Instruct_full_dialog_top_5"]["category_counts"]

cums = data["meta-llama/Llama-3.1-8B-Instruct_full_dialog_top_5"][
    "cum_accuracy_by_category"
]

# Compute average F1 per category (cumulative / number of questions)
avg_f1 = [cums[str(cat)] / counts[str(cat)] for cat in range(1, 6)]

# --------------------------------------------------------------------
# Plotting – same style, but y-axis fixed to 0–0.8
# --------------------------------------------------------------------

categories = [
    "1\n(Factual)",
    "2\n(Temporal)",
    "3\n(Inference)",
    "4\n(Multi-hop)",
    "5\n(Adversarial)",
]

colors = ["#A8E6CF", "#FFD3B6", "#FFAAA5", "#FF8B94", "#C7CEEA"]

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(categories, avg_f1, color=colors, edgecolor="grey", linewidth=0.8)

ax.set_xlabel("Question Category", fontsize=12)
ax.set_ylabel("Average F1 Score", fontsize=12)

ax.set_title(
    "Per Category Performance of Llama 3.1 8B (Full Dialog RAG, Top-5)",
    fontsize=14,
    pad=20,
)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

# Fixed y-axis scale
ax.set_ylim(0, 0.8)

plt.tight_layout()
# plt.savefig("category_performance.pdf", dpi=300, bbox_inches="tight")
plt.show()
