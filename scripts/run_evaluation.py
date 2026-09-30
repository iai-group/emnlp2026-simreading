"""Evaluates simulators against the ground truth, from aggregates.

Reads the shipped aggregated data (no individual responses, no LLM calls)
and reproduces the paper's main results table. For each method under
``data/aggregates/simulation/`` it loads the per-run question-stats and
score-histogram CSVs, computes the metric set with
``simreading.aggregates.evaluate_aggregates``, and reports mean +/- std
across runs.

Ground truth is the held-out *test* split; the "Real students (Held-out)"
reference row evaluates the *train* split against test.

A freshly-run simulator (see ``run_simulation.py``) has no score histogram,
so its JSD is reported as ``nan``; all other metrics are exact.

Usage:
    python scripts/run_evaluation.py            # summary table
    python scripts/run_evaluation.py --latex    # LaTeX rows (tab:results)
"""

import argparse
import glob
import os
import statistics

from simreading.aggregates import (
    evaluate_aggregates,
    load_question_stats,
    load_score_hist,
)
from simreading.experiments import (
    label_for_short_name,
    load_models,
    parse_method_dirname,
)

HUMAN_DIR = "data/aggregates/human"
SIM_DIR = "data/aggregates/simulation"

_MODELS = load_models()
_METRIC_KEYS = ("sim_acc", "gap", "ece", "pearson", "spearman", "dist_jsd")

_VARIANT_LABELS = {
    "llm_baseline": "baseline",
    "llm_baseline_no": "baseline (Bokmal)",
    "llm_lowpersona": "low-persona",
    "cbus_spr_dropout": "CBUS-SPR dropout",
    "answer_noising": "answer-noising",
}


def _label_parts(method_dir: str) -> tuple[str, str]:
    parsed = parse_method_dirname(method_dir)
    if parsed is None:
        return (os.path.basename(method_dir.rstrip("/")), "")
    simulator, model_short, capacity = parsed
    if simulator == "random":
        return ("Random", "")
    label = label_for_short_name(model_short, _MODELS)
    if simulator == "cbus_ts":
        return (label, f"CBUS-TS (C={capacity})")
    if simulator == "cbus_spr":
        return (label, f"CBUS-SPR (C={capacity})")
    return (label, _VARIANT_LABELS.get(simulator, simulator))


# ---------------------------------------------------------------------------
# Loading + evaluating aggregate runs
# ---------------------------------------------------------------------------

def _load_pred(qs_path: str):
    """Loads (question_stats, score_hist-or-None) for one run/reference."""
    qs = load_question_stats(qs_path)
    hist_path = qs_path.replace("_question_stats.csv", "_score_hist.csv")
    if hist_path == qs_path:  # reference files: question_stats_train.csv
        hist_path = qs_path.replace("question_stats_", "score_hist_")
    hist = load_score_hist(hist_path) if os.path.exists(hist_path) else None
    return qs, hist


def _evaluate_method(real_qs, real_hist, run_files: list[str]) -> list[dict]:
    runs = []
    for rf in run_files:
        sim_qs, sim_hist = _load_pred(rf)
        runs.append(evaluate_aggregates(real_qs, real_hist, sim_qs, sim_hist))
    return runs


def _aggregate(runs: list[dict]) -> dict:
    agg = {"n_runs": len(runs)}
    for key in _METRIC_KEYS:
        values = [r[key] for r in runs]
        agg[f"{key}_mean"] = statistics.mean(values)
        agg[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return agg


_SIMULATOR_ORDER = {
    "llm_baseline": 0, "llm_baseline_no": 1, "llm_lowpersona": 2,
    "answer_noising": 3, "cbus_spr_dropout": 4, "cbus_ts": 5, "cbus_spr": 6,
}


def _method_sort_key(name: str) -> tuple:
    parsed = parse_method_dirname(name)
    if parsed is None:
        return (2, name)
    simulator, model_short, capacity = parsed
    if simulator == "random":
        return (0,)
    return (1, model_short or "", _SIMULATOR_ORDER.get(simulator, 9), capacity or 0)


def _discover_methods() -> list[tuple[str, str, list[str]]]:
    methods = []
    if not os.path.isdir(SIM_DIR):
        return methods
    names = sorted(
        (n for n in os.listdir(SIM_DIR) if not n.startswith("_")),
        key=_method_sort_key,
    )
    for name in names:
        d = os.path.join(SIM_DIR, name)
        if not os.path.isdir(d):
            continue
        run_files = sorted(glob.glob(os.path.join(d, "r*_question_stats.csv")))
        if run_files:
            llm, variant = _label_parts(name)
            methods.append((llm, variant, run_files))
    return methods


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    headers = {
        "llm": "LLM", "variant": "Variant", "sim_acc": "Accuracy",
        "gap": "AbsPerfGap", "ece": "ECE", "pearson": "Pearson",
        "spearman": "Spearman", "dist_jsd": "Score JSD", "n_runs": "Runs",
    }
    col_keys = ("llm", "variant") + _METRIC_KEYS + ("n_runs",)

    cells = []
    for r in results:
        n = r["n_runs"]
        row = {"llm": r["llm"], "variant": r["variant"], "n_runs": f"({n})"}
        for key in _METRIC_KEYS:
            mean, std = r[f"{key}_mean"], r[f"{key}_std"]
            row[key] = f"{mean:.3f}" if n == 1 else f"{mean:.3f}+/-{std:.3f}"
        cells.append(row)

    widths = {
        k: max(len(headers[k]), max((len(str(c[k])) for c in cells), default=0))
        for k in col_keys
    }

    def fmt(v):
        return "  ".join(
            f"{str(v[k]):<{widths[k]}}" if k in ("llm", "variant")
            else f"{str(v[k]):>{widths[k]}}"
            for k in col_keys
        )

    line = fmt(headers)
    print("\n" + "=" * len(line))
    print("Summary  (GT = held-out test split)")
    print("=" * len(line))
    print(line)
    print("-" * len(line))
    prev = None
    for c in cells:
        disp = dict(c)
        if c["llm"] == prev:
            disp["llm"] = ""
        else:
            prev = c["llm"]
        print(fmt(disp))
    print("=" * len(line))


def _std_tex(std: float) -> str:
    return "{:.3f}".format(std).lstrip("0")


def _print_latex(results: list[dict]) -> None:
    """Emits rows for tab:results: gap, JSD, ECE, Pearson, Spearman."""
    order = ("gap", "dist_jsd", "ece", "pearson", "spearman")
    print("% Abs.Gap & JSD & ECE & Pearson & Spearman")
    for r in results:
        cells = " & ".join(
            f"{r[f'{k}_mean']:.3f} & \\stdev{{{_std_tex(r[f'{k}_std'])}}}"
            for k in order
        )
        variant = r["variant"] or ""
        print(f"    {r['llm']} & {variant} & {cells} \\\\")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latex", action="store_true", help="Emit LaTeX rows.")
    args = ap.parse_args()

    real_qs = load_question_stats(os.path.join(HUMAN_DIR, "question_stats_test.csv"))
    real_hist = load_score_hist(os.path.join(HUMAN_DIR, "score_hist_test.csv"))

    results = []

    # Reference: held-out train split vs test.
    ref_qs = os.path.join(HUMAN_DIR, "question_stats_train.csv")
    if os.path.exists(ref_qs):
        runs = _evaluate_method(real_qs, real_hist, [ref_qs])
        results.append({"llm": "Real students (Held-out)", "variant": "",
                        **_aggregate(runs)})

    for llm, variant, run_files in _discover_methods():
        runs = _evaluate_method(real_qs, real_hist, run_files)
        results.append({"llm": llm, "variant": variant, **_aggregate(runs)})

    if args.latex:
        _print_latex(results)
    else:
        _print_summary(results)


if __name__ == "__main__":
    main()
