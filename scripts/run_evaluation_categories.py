"""Per-category evaluation, from aggregates (Appendix tab:category_results).

Reproduces the reading-comprehension category breakdown for the two
backbones shown in the paper (Llama-3.3-70B and Gemini-3.5-Flash-Lite).
Questions are grouped following the paper:

  - ``Locate``              -> category ``Locate``
  - ``Interpret & Reflect`` -> ``Interpret and Connect`` +
                               ``Reflect and Evaluate``

Item-centric metrics (Abs. Gap, ECE, Pearson, Spearman) come from filtering
the question-stats aggregate by category; JSD comes from the per-category
score histograms shipped for these methods. Categories are read from
``data/questions.json``.

Usage:
    python scripts/run_evaluation_categories.py
    python scripts/run_evaluation_categories.py --latex
"""

import argparse
import glob
import os
import statistics

from simreading import Questions
from simreading.aggregates import (
    absolute_performance_gap,
    difficulty_correlation,
    distributional_alignment,
    expected_calibration_error,
    load_question_stats,
    load_score_hist,
)
from simreading.experiments import cbus_default, method_dirname

HUMAN_DIR = "data/aggregates/human"
SIM_DIR = "data/aggregates/simulation"

BACKBONES = [
    ("Llama-3.3-70b-Instruct", "llama-3.3-70b-instruct"),
    ("Gemini-3.5-Flash-Lite", "gemini-3.5-flash-lite"),
]

# Display name -> (raw category set, file-safe label used in score_hist files).
GROUPS = {
    "Locate": ({"Locate"}, "Locate"),
    "Interpret & Reflect": (
        {"Interpret and Connect", "Reflect and Evaluate"},
        "InterpretReflect",
    ),
}

_METRICS = [
    ("gap", "Abs. Gap"), ("dist_jsd", "JSD"), ("ece", "ECE"),
    ("pearson", "Pearson"), ("spearman", "Spearman"),
]


def _variant_dirs(short: str) -> list[tuple[str, str]]:
    return [
        ("baseline", method_dirname("llm_baseline", short)),
        ("CBUS-TS", method_dirname("cbus_ts", short, cbus_default("ts"))),
        ("CBUS-SPR", method_dirname("cbus_spr", short, cbus_default("spr"))),
    ]


def _filter_qs(qs: dict, catmap: dict, cats: set) -> dict:
    return {(t, q): v for (t, q), v in qs.items() if catmap.get(q) in cats}


def _metrics_from_run(real_qs, real_hist, sim_qs, sim_hist) -> dict:
    corr = difficulty_correlation(real_qs, sim_qs)
    return {
        "gap": absolute_performance_gap(real_qs, sim_qs),
        "ece": expected_calibration_error(real_qs, sim_qs)["ece"],
        "pearson": corr["pearson"],
        "spearman": corr["spearman"],
        "dist_jsd": distributional_alignment(real_hist, sim_hist),
    }


def _mean_metrics(real_qs, real_hist, method: str, label: str,
                  catmap: dict, cats: set) -> dict | None:
    d = os.path.join(SIM_DIR, method)
    run_files = sorted(glob.glob(os.path.join(d, "r*_question_stats.csv")))
    per = {k: [] for k, _ in _METRICS}
    for rf in run_files:
        run = os.path.basename(rf).split("_")[0]  # r1
        hist_path = os.path.join(d, f"{run}_score_hist_{label}.csv")
        if not os.path.exists(hist_path):
            continue
        sim_qs = _filter_qs(load_question_stats(rf), catmap, cats)
        sim_hist = load_score_hist(hist_path)
        m = _metrics_from_run(real_qs, real_hist, sim_qs, sim_hist)
        for k, _ in _METRICS:
            per[k].append(m[k])
    if not per["gap"]:
        return None
    return {k: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
            for k, v in per.items()}


def _std_tex(std: float) -> str:
    return "{:.3f}".format(std).lstrip("0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    questions = Questions()
    catmap = {q.id: q.category for q in questions.get_all_questions()}
    real_qs_all = load_question_stats(
        os.path.join(HUMAN_DIR, "question_stats_test.csv")
    )

    for gname, (cats, flabel) in GROUPS.items():
        real_qs = _filter_qs(real_qs_all, catmap, cats)
        real_hist = load_score_hist(
            os.path.join(HUMAN_DIR, f"score_hist_test_{flabel}.csv")
        )
        if not args.latex:
            n_q = sum(1 for q in questions.get_all_questions() if q.category in cats)
            print(f"\n=== {gname} (n_questions={n_q}) ===")
            print(f"  {'backbone':24s} {'variant':9s} "
                  + " ".join(f"{lbl:>9}" for _, lbl in _METRICS))
        for blabel, short in BACKBONES:
            for vlabel, method in _variant_dirs(short):
                m = _mean_metrics(
                    real_qs, real_hist, method, flabel, catmap, cats
                )
                if args.latex:
                    if m is None:
                        cells = " & ".join("XXX &" for _ in _METRICS)
                    else:
                        cells = " & ".join(
                            f"{m[k][0]:.3f} & \\stdev{{{_std_tex(m[k][1])}}}"
                            for k, _ in _METRICS
                        )
                    print(f"    {blabel} & {vlabel} & {cells} \\\\  % {gname}")
                else:
                    vals = ("  (missing)" if m is None else
                            " ".join(f"{m[k][0]:.3f}_{m[k][1]:.3f}"
                                     for k, _ in _METRICS))
                    print(f"  {blabel:24s} {vlabel:9s} {vals}")


if __name__ == "__main__":
    main()
