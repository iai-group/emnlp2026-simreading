"""Plots CBUS capacity (C) sensitivity curves (from aggregates).

For the two backbones with a full capacity sweep (Llama-3.3-70B and
Gemini-3.5-Flash-Lite), evaluates every available C for both strategies
(SPR, TS) against the ground-truth test split and plots student-centric
JSD and item-centric Pearson as a function of C. The cognitively-motivated
operating points (SPR C=4, TS C=2) are marked with a vertical line, and each
backbone's persona baseline is a horizontal reference line.

Produces ``figures/capacity_sweep.pdf``.
"""

import glob
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from simreading.aggregates import (
    difficulty_correlation,
    distributional_alignment,
    load_question_stats,
    load_score_hist,
)
from simreading.plot_style import apply_paper_style, LEGEND_SIZE

_HUMAN = "data/aggregates/human"
_SIM = "data/aggregates/simulation"
_OUT = "figures/capacity_sweep.pdf"

_MODELS = [
    ("llama-3.3-70b-instruct", "Llama-3.3-70B", "dodgerblue", "o"),
    ("gemini-3.5-flash-lite", "Gemini-3.5-Flash-Lite", "tomato", "s"),
]
_FIXED_C = {"spr": 4, "ts": 2}
_STRATEGIES = [("spr", "CBUS-SPR"), ("ts", "CBUS-TS")]


def _eval_dir(d, real_qs, real_hist):
    """Mean (JSD, Pearson) over the runs in a method dir."""
    jsds, pears = [], []
    for rf in sorted(glob.glob(os.path.join(d, "r*_question_stats.csv"))):
        run = os.path.basename(rf).split("_")[0]
        hist = os.path.join(d, f"{run}_score_hist.csv")
        sim_qs = load_question_stats(rf)
        jsds.append(distributional_alignment(real_hist, load_score_hist(hist)))
        pears.append(difficulty_correlation(real_qs, sim_qs)["pearson"])
    if not jsds:
        return None
    return sum(jsds) / len(jsds), sum(pears) / len(pears)


def _sweep_points(strategy, model_short, real_qs, real_hist):
    points = {}
    for d in glob.glob(os.path.join(_SIM, f"cbus_{strategy}-c*-{model_short}")):
        cap = os.path.basename(d).split("-c", 1)[1].split("-", 1)[0]
        if not cap.isdigit():
            continue
        res = _eval_dir(d, real_qs, real_hist)
        if res:
            points[int(cap)] = res
    cs = sorted(points)
    return cs, [points[c][0] for c in cs], [points[c][1] for c in cs]


def main() -> None:
    real_qs = load_question_stats(os.path.join(_HUMAN, "question_stats_test.csv"))
    real_hist = load_score_hist(os.path.join(_HUMAN, "score_hist_test.csv"))

    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="col")
    for col, (strat, strat_label) in enumerate(_STRATEGIES):
        ax_jsd, ax_p = axes[0][col], axes[1][col]
        for model_short, label, color, marker in _MODELS:
            cs, jsd, pear = _sweep_points(strat, model_short, real_qs, real_hist)
            if not cs:
                continue
            ax_jsd.plot(cs, jsd, color=color, marker=marker, label=label)
            ax_p.plot(cs, pear, color=color, marker=marker, label=label)
            base = _eval_dir(
                os.path.join(_SIM, f"llm_baseline-{model_short}"),
                real_qs, real_hist,
            )
            if base:
                ax_jsd.axhline(base[0], color=color, ls=":", lw=1.5, zorder=0)
                ax_p.axhline(base[1], color=color, ls=":", lw=1.5, zorder=0)
        for ax in (ax_jsd, ax_p):
            ax.axvline(_FIXED_C[strat], color="gray", ls="--", lw=1, zorder=0)
        ax_jsd.set_title(strat_label)
        ax_p.set_xlabel("Capacity $C$")
    axes[0][0].set_ylabel("Score JSD ($\\downarrow$)")
    axes[1][0].set_ylabel("Pearson $\\rho$ ($\\uparrow$)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="gray", ls=":", lw=1.5))
    labels.append("Persona baseline")
    axes[0][0].legend(handles, labels, fontsize=LEGEND_SIZE, loc="best")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(_OUT, bbox_inches="tight")
    print(f"Wrote {_OUT}")


if __name__ == "__main__":
    main()
