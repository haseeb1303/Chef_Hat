import json
import matplotlib.pyplot as plt

# Files produced by log_results.py
files = {
    "Baseline (Random)": "task2_rl/output/dqn_vs_random.json",
    "Generative (Untuned)": "task2_rl/output/dqn_vs_generative.json",
    "Generative (Tuned)": "task2_rl/output/dqn_vs_generative_tuned.json"
}

labels = []
dqn_scores = []

for label, path in files.items():
    try:
        with open(path, "r") as f:
            data = json.load(f)
            labels.append(label)
            dqn_scores.append(data["scores"].get("DQN", 0))
    except FileNotFoundError:
        print(f"[WARN] Missing file: {path}")

plt.figure()
plt.bar(labels, dqn_scores)
plt.ylabel("DQN Score")
plt.title("DQN Performance Across Experiments")
plt.tight_layout()
plt.savefig("task2_rl/plot_results/dqn_score_comparison.png", dpi=300)
plt.show()