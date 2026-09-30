"""Plots distributions of empirical question difficulty (from aggregates).

Difficulty is the mean correctness rate per ``(text_id, question_id)`` across
all real students, read from the shipped question-stats aggregates (train +
test combined). Items with fewer than 5 responses are skipped. Produces:

    figures/question_difficulty_by_type.pdf      — by question type
    figures/question_difficulty_by_category.pdf  — by question category
"""

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import gaussian_kde

from simreading import Questions
from simreading.aggregates import (
    MIN_RESPONSES_PER_QUESTION,
    NUM_BINS,
    load_question_stats,
)
from simreading.plot_style import apply_paper_style, LEGEND_SIZE

HUMAN_DIR = "data/aggregates/human"
FIGURES_DIR = "figures"
_COLORS = ["dodgerblue", "tomato", "forestgreen", "goldenrod", "purple", "gray"]


def _combined_stats() -> dict:
    """Sums train + test question stats into {(t, q): (n_correct, n_total)}."""
    combined: dict = defaultdict(lambda: [0, 0])
    for split in ("train", "test"):
        qs = load_question_stats(os.path.join(HUMAN_DIR, f"question_stats_{split}.csv"))
        for key, (nc, ni) in qs.items():
            combined[key][0] += nc
            combined[key][1] += nc + ni
    return combined


def _collect(questions: Questions):
    by_type: dict[str, list[float]] = defaultdict(list)
    by_category: dict = defaultdict(list)
    for (text_id, q_id), (nc, n) in _combined_stats().items():
        if n < MIN_RESPONSES_PER_QUESTION:
            continue
        q = questions.get_question(q_id)
        if q is None:
            continue
        by_type[q.type].append(nc / n)
        by_category[q.category].append(nc / n)
    return dict(by_type), dict(by_category)


def _plot_kde(series, title, output_path, order=None) -> None:
    keys = order if order is not None else sorted(series.keys())
    apply_paper_style()
    plt.figure(figsize=(10, 6))
    bin_width = 1.0 / NUM_BINS
    x = np.linspace(0, 1, 200)
    for i, key in enumerate(keys):
        values = series.get(key, [])
        if len(values) < 2:
            print(f"  (skipping {key!r}: {len(values)} point(s))")
            continue
        color = _COLORS[i % len(_COLORS)]
        y = gaussian_kde(values)(x) * bin_width * 100
        label = (key if key is not None else "(no category)") + f" (n={len(values)})"
        plt.fill_between(x, y, alpha=0.25, color=color)
        plt.plot(x, y, color=color, linewidth=2, label=label)
    plt.title(title)
    plt.xlabel("Question difficulty (mean accuracy)")
    plt.ylabel("Percentage of questions")
    plt.gca().yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    plt.legend(loc="upper left", fontsize=LEGEND_SIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.close()


def main() -> None:
    questions = Questions()
    by_type, by_category = _collect(questions)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    type_order = ["multiChoice", "trueOrFalse", "checkboxes"]
    _plot_kde(
        by_type, "Question difficulty by type",
        os.path.join(FIGURES_DIR, "question_difficulty_by_type.pdf"),
        order=[t for t in type_order if t in by_type],
    )

    category_order = ["Locate", "Interpret and Connect", "Reflect and Evaluate"]
    keys = [c for c in category_order if c in by_category]
    if None in by_category:
        keys.append(None)
    _plot_kde(
        by_category, "Question difficulty by category",
        os.path.join(FIGURES_DIR, "question_difficulty_by_category.pdf"),
        order=keys,
    )


if __name__ == "__main__":
    main()
