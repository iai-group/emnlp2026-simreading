"""Plots the student performance distribution comparison (Figure 1).

Reads the shipped 20-bin per-student score histograms and overlays them.
Unlike the paper's smooth KDE (which needs individual student scores), the
release renders the histogram directly, which reproduces the JSD exactly.

Default sources reproduce Figure 1: real students, random guessing, and the
Gemini-3.5-Flash-Lite / Gemini-3.7-Flash persona baselines (sequential green
palette: the darker, more capable model reads as further from human).

Usage:
    python scripts/plot_comparison.py
    python scripts/plot_comparison.py path/to/score_hist.csv:Label:color ...
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from simreading.aggregates import NUM_BINS, load_score_hist
from simreading.plot_style import apply_paper_style, LEGEND_SIZE

_HUMAN = "data/aggregates/human"
_SIM = "data/aggregates/simulation"

# (score_hist path, label, color)
_DEFAULT_SOURCES = [
    (f"{_HUMAN}/score_hist_test.csv", "Real students", "dodgerblue"),
    (f"{_SIM}/random/r1_score_hist.csv", "Random guessing", "tomato"),
    (f"{_SIM}/llm_baseline-gemini-3.5-flash-lite/r1_score_hist.csv",
     "Gemini-3.5-Flash-Lite", "#74c476"),
    (f"{_SIM}/llm_baseline-gemini-3.7-flash/r1_score_hist.csv",
     "Gemini-3.7-Flash", "#006d2c"),
]
_COLORS = ["dodgerblue", "tomato", "forestgreen", "goldenrod", "purple", "gray"]


def plot_comparison(sources, output_path=None, y_max=None) -> None:
    edges = np.linspace(0, 1, NUM_BINS + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0

    apply_paper_style()
    plt.figure(figsize=(10, 6))

    for i, src in enumerate(sources):
        path, label = src[0], src[1]
        color = src[2] if len(src) > 2 else _COLORS[i % len(_COLORS)]
        if not os.path.exists(path):
            print(f"WARN: missing {path}, skipping.")
            continue
        counts = load_score_hist(path)
        pct = counts / counts.sum() * 100
        plt.fill_between(mids, pct, alpha=0.25, color=color, step="mid")
        plt.plot(mids, pct, color=color, linewidth=2, label=label, marker="o",
                 markersize=3)

    plt.xlabel("Fraction of Correct Answers")
    plt.ylabel("Percentage of Students")
    plt.gca().yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    if y_max is not None:
        plt.ylim(0, y_max)
    plt.legend(loc="upper left", fontsize=LEGEND_SIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    out = output_path or "figures/student_performance_kde.pdf"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out)
    print(f"Plot saved to: {out}")
    plt.close()


def _parse_source(spec: str):
    parts = spec.split(":")
    return tuple(parts) if len(parts) >= 2 else (spec, os.path.basename(spec))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", nargs="*", help="score_hist.csv:Label[:color]")
    ap.add_argument("--output", default=None)
    ap.add_argument("--y-max", type=float, default=None)
    args = ap.parse_args()

    sources = [_parse_source(s) for s in args.sources] or _DEFAULT_SOURCES
    plot_comparison(sources, output_path=args.output, y_max=args.y_max)
