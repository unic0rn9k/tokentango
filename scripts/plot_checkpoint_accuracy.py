import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv("checkpoint_test_results.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    df["datetime"],
    df["checkpoint_accuracy"],
    marker="o",
    linewidth=2,
    markersize=8,
    color="blue",
    label="Saved Accuracy",
)

ax.plot(
    df["datetime"],
    df["test_accuracy"],
    marker="s",
    linewidth=2,
    markersize=8,
    color="red",
    label="Test Accuracy (0.8 frac)",
)

ax.set_title(
    "Model Accuracy Over Time (>80% Checkpoints)", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Checkpoint DateTime", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("checkpoint_accuracy_plot.png", dpi=150, bbox_inches="tight")
print("Plot saved to checkpoint_accuracy_plot.png")
plt.close()
