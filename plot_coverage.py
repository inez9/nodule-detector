# plot_coverage.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/batch_summary.csv")

# Extract node ID (everything up to "_L")
df["node"] = df["image"].str.extract(r"(Node\d+)")
plt.figure(figsize=(10,5))
for node, sub in df.groupby("node"):
    plt.bar(sub["image"], sub["coverage_pct"], label=node)

plt.xticks(rotation=45, ha="right", fontsize=8)
plt.ylabel("Coverage (%)")
plt.title("Estimated nodule coverage per image")
plt.legend()
plt.tight_layout()
plt.savefig("data/processed/coverage_bar.png", dpi=300)
plt.show()

