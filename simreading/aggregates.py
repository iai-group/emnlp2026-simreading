"""Evaluation metrics computed from aggregated data.

The public release ships only aggregates, never individual responses:

  * **question stats** — per ``(text_id, question_id)`` counts of correct
    and incorrect answers. These reproduce Abs. Gap, ECE, Pearson, and
    Spearman *exactly* (the source metrics are all functions of per-item
    success rates and response-weighted overall accuracy).
  * **score histogram** — a 20-bin histogram of per-student
    fraction-correct. This reproduces the JSD *exactly*, because the
    original JSD metric itself bins per-student fractions into 20 bins
    before comparing distributions.
"""

from __future__ import annotations

import csv

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr

# Must match the original metric implementation.
NUM_BINS = 20
MIN_RESPONSES_PER_QUESTION = 5
ECE_BINS = 6

QuestionStats = dict  # {(text_id, question_id): (n_correct, n_incorrect)}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_question_stats(path: str) -> QuestionStats:
    """Loads a ``question_stats`` CSV into ``{(t, q): (n_correct, n_incorrect)}``."""
    stats: QuestionStats = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stats[(row["sanity_text_id"], row["question_id"])] = (
                int(row["n_correct"]),
                int(row["n_incorrect"]),
            )
    return stats


def load_score_hist(path: str) -> np.ndarray:
    """Loads a ``score_hist`` CSV into a length-20 count array."""
    with open(path, encoding="utf-8") as f:
        return np.array([int(r["n_students"]) for r in csv.DictReader(f)])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def overall_accuracy(qs: QuestionStats) -> float:
    """Response-weighted accuracy: ``sum(correct) / sum(total)``."""
    n_correct = sum(c for c, _ in qs.values())
    n_total = sum(c + i for c, i in qs.values())
    return n_correct / n_total if n_total else 0.0


def per_item_rates(qs: QuestionStats) -> dict:
    """Per-item success rate ``n_correct / n_total`` (skips empty items)."""
    return {k: c / (c + i) for k, (c, i) in qs.items() if (c + i) > 0}


def absolute_performance_gap(real_qs: QuestionStats, sim_qs: QuestionStats) -> float:
    """``|overall_accuracy(real) - overall_accuracy(sim)|``."""
    return abs(overall_accuracy(real_qs) - overall_accuracy(sim_qs))


def _aligned_pairs(real_qs, sim_qs):
    """(real_rate, sim_rate) pairs over common items with >= MIN real responses."""
    real_rates = per_item_rates(real_qs)
    sim_rates = per_item_rates(sim_qs)
    pairs = []
    for key in set(real_rates) & set(sim_rates):
        if sum(real_qs[key]) < MIN_RESPONSES_PER_QUESTION:
            continue
        pairs.append((real_rates[key], sim_rates[key]))
    return pairs


def difficulty_correlation(real_qs, sim_qs) -> dict:
    """Pearson + Spearman of per-item difficulty over aligned items."""
    pairs = _aligned_pairs(real_qs, sim_qs)
    if len(pairs) < 2:
        return {
            "pearson": float("nan"),
            "spearman": float("nan"),
            "n_items": len(pairs),
        }
    real_vec = [a for a, _ in pairs]
    sim_vec = [b for _, b in pairs]
    return {
        "pearson": float(pearsonr(real_vec, sim_vec)[0]),
        "spearman": float(spearmanr(real_vec, sim_vec)[0]),
        "n_items": len(pairs),
    }


def expected_calibration_error(real_qs, sim_qs, n_bins: int = ECE_BINS) -> dict:
    """ECE with items binned by *human* success rate into ``n_bins`` bins."""
    pairs = _aligned_pairs(real_qs, sim_qs)
    n_items = len(pairs)
    if n_items == 0:
        return {"ece": float("nan"), "n_bins": n_bins, "n_items": 0}

    buckets = [[] for _ in range(n_bins)]
    for human_r, sim_r in pairs:
        idx = min(int(human_r * n_bins), n_bins - 1)
        buckets[idx].append((human_r, sim_r))

    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        w = len(bucket) / n_items
        s_human = sum(h for h, _ in bucket) / len(bucket)
        s_sim = sum(s for _, s in bucket) / len(bucket)
        ece += w * abs(s_human - s_sim)
    return {"ece": float(ece), "n_bins": n_bins, "n_items": n_items}


def distributional_alignment(real_hist: np.ndarray, sim_hist: np.ndarray) -> float:
    """JSD (base 2, squared) between two 20-bin student-score histograms."""
    real_dist = real_hist / real_hist.sum()
    sim_dist = sim_hist / sim_hist.sum()
    return float(jensenshannon(real_dist, sim_dist, base=2) ** 2)


def evaluate_aggregates(
    real_qs: QuestionStats,
    real_hist: np.ndarray | None,
    sim_qs: QuestionStats,
    sim_hist: np.ndarray | None,
) -> dict:
    """Full metric set from aggregates.

    Returns a dict with keys ``real_acc``, ``sim_acc``, ``gap``, ``ece``,
    ``pearson``, ``spearman``, ``dist_jsd``. ``dist_jsd`` is ``nan`` when
    either histogram is missing (e.g. a freshly-run simulator, which has no
    per-student score distribution — see RELEASE_PLAN.md).
    """
    corr = difficulty_correlation(real_qs, sim_qs)
    jsd = float("nan")
    if real_hist is not None and sim_hist is not None:
        jsd = distributional_alignment(real_hist, sim_hist)
    return {
        "real_acc": overall_accuracy(real_qs),
        "sim_acc": overall_accuracy(sim_qs),
        "gap": absolute_performance_gap(real_qs, sim_qs),
        "ece": expected_calibration_error(real_qs, sim_qs)["ece"],
        "pearson": corr["pearson"],
        "spearman": corr["spearman"],
        "dist_jsd": jsd,
    }
