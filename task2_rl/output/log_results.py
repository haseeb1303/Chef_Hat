import json
import os
from datetime import datetime

def save_results(experiment_name, final_scores):
    """
    Save experiment results to JSON for reproducibility
    """
    os.makedirs("task2_rl/output", exist_ok=True)

    result = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "scores": final_scores
    }

    filename = f"task2_rl/output/{experiment_name}.json"

    with open(filename, "w") as f:
        json.dump(result, f, indent=4)

    print(f"[LOG] Results saved to {filename}")
