"""Plots text popularity: unique students per text, ranked (from aggregates).

Reads the shipped per-text reader counts (train + test combined) and plots
them in descending order. Produces ``figures/texts_by_popularity.pdf``.
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from simreading.plot_style import apply_paper_style

HUMAN_DIR = "data/aggregates/human"
_OUT = "figures/texts_by_popularity.pdf"


def _combined_reader_counts() -> dict:
    counts: dict = defaultdict(int)
    for split in ("train", "test"):
        path = os.path.join(HUMAN_DIR, f"text_reader_counts_{split}.csv")
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                counts[row["sanity_text_id"]] += int(row["n_readers"])
    return counts


def main() -> None:
    counts = sorted(_combined_reader_counts().values(), reverse=True)
    os.makedirs("figures", exist_ok=True)
    apply_paper_style()
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(counts)), counts, marker="s", color="orange",
             linestyle="-", markersize=2)
    plt.title("Text Popularity (Unique Students per Text)")
    plt.xlabel("Texts (ordered by number of students)")
    plt.ylabel("Number of Students")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(_OUT)
    print(f"Plot saved to: {_OUT}")
    plt.close()


if __name__ == "__main__":
    main()
