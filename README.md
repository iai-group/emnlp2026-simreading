# "Act Like a 5th Grader" is Not Enough: Bounding Knowledge in LLM-Based User Simulators

This repository **will provide** resources developed within the following article:

> Krisztian Balog and Arild Michel Bakken. **"Act Like a 5th Grader" is Not Enough: Bounding Knowledge in LLM-Based User Simulators.** In: Findings of the Association for Computational Linguistics: EMNLP 2026. Association for Computational Linguistics. 2026.

## Summary

Large language models (LLMs) are increasingly used to simulate human behavior but frequently fail to exhibit realistic cognitive constraints, suffering from a "superhuman bias." Using a dataset of over 71,000 reading comprehension responses from 2,359 primary-school students (grades 4–6), we demonstrate that standard persona prompting yields near-perfect, deterministic performance, failing to capture the natural variance of developing readers. To address this, we introduce the Cognitively Bounded User Simulator (CBUS), an architectural framework that explicitly models the restricted working memory of young readers through an episodic bottleneck. Within this framework, we formalize two distinct test-taking strategies to emulate different reading behaviors. Our evaluation shows that explicitly modeling cognitive bounds significantly narrows the simulation gap across multiple LLM backbones, demonstrating that enforcing architectural constraints is more effective for high-fidelity simulation than simply scaling raw model capabilities.

## What this repository contains

This is an **aggregated** release. Individual student responses are not
published; instead we ship question-level counts and binned score
distributions that reproduce every number and figure in the paper.

  * [`data/questions.json`](data/questions.json) — item bank: 156 texts with their questions, answer options, gold answers, and categories.
  * [`data/question_categories.csv`](data/question_categories.csv) — `question_id` → reading-comprehension category.
  * [`data/aggregates/human/`](data/aggregates/human/) — human aggregates, per train/test split:
    - `question_stats_{train,test}.csv` — per `(text, question)`: `n_correct`, `n_incorrect`.
    - `score_hist_{train,test}.csv` — 20-bin histogram of per-student accuracy.
    - `score_hist_test_{Locate,InterpretReflect}.csv` — per-category (for category JSD).
    - `text_reader_counts_{train,test}.csv` — unique students per text.
  * [`data/aggregates/simulation/`](data/aggregates/simulation/) — precomputed simulator aggregates, per `<method>` and run:
    - `r{run}_question_stats.csv` — simulator per-item counts.
    - `r{run}_score_hist.csv` — simulator score histogram.
  * [`simreading/`](simreading/) — Python package (data model, CBUS simulator, metrics).
  * [`scripts/`](scripts/) — evaluation, plotting, and simulation entry points.
  * [`models.yaml`](models.yaml) — backbone list and CBUS capacity defaults/sweeps.

The `test` split is the ground truth; `train` is the held-out human reference.

**Why aggregates reproduce the paper exactly.** Abs. Gap, ECE, Pearson, and
Spearman are functions of per-item success rates and response-weighted
accuracy, so the question-level counts reproduce them exactly. JSD is computed
by binning per-student accuracy into 20 bins, so the shipped 20-bin histograms
reproduce it exactly as well.

## Installation

```bash
pip install -e .          # or: pip install -r requirements.txt
```

Python 3.9+. Reproducing the paper (below) needs no API keys.

## Reproduce the paper (offline, no LLM calls)

```bash
python scripts/run_evaluation.py                # Table 1 (main results)
python scripts/run_evaluation.py --latex        # ... as LaTeX rows
python scripts/run_evaluation_categories.py     # Appendix category breakdown
python scripts/run_evaluation_categories.py --latex

python scripts/plot_comparison.py               # Figure 1 (score distributions)
python scripts/plot_capacity_sweep.py           # Capacity (C) sensitivity
python scripts/plot_question_difficulty.py      # Difficulty by type / category
python scripts/plot_texts_by_popularity.py      # Text popularity
```

Figures are written to `figures/`. Figure 1 is rendered from the 20-bin score
histograms (which reproduce the exact JSD) rather than the paper's smoothed
KDE, so it appears slightly more angular but is otherwise identical.

## Run a new simulator

You can run the persona baseline or a CBUS variant with any LLM over the item
bank. Copy the config and set your provider/key:

```bash
cp llm_config.example.yaml llm_config.yaml      # then edit it

python scripts/run_simulation.py --simulator cbus_spr --model <provider/model>
python scripts/run_simulation.py --simulator random --limit 5   # quick offline check
python scripts/run_evaluation.py                # scores the new run
```

Because no per-student exposure is released, a new run uses *synthetic readers*
(each reads one text, drawn from the shipped per-text reader counts). This
reproduces the **item-centric** metrics (Abs. Gap, ECE, Pearson, Spearman) up
to sampling noise, but **not** JSD or Figure 1 — those need per-student exposure
and are available only from the shipped precomputed aggregates. Simulators:
`random`, `llm_baseline`, `llm_baseline_no`, `llm_lowpersona`, `cbus_ts`,
`cbus_spr`, `cbus_spr_dropout`.

## Data availability and license

The reading texts and questions are third-party copyrighted material,
redistributed here for research reproducibility. Student responses are released
only in aggregated form (no individual-level data). **License: to be finalized
before publication** — see the paper and `RELEASE_PLAN.md`.

## Citation

If you use the resources presented in this repository, please cite:

```
@inproceedings{Balog:2026:EMNLP,
  author =    {Balog, Krisztian and Bakken, Arild Michel},
  title =     {``Act Like a 5th Grader'' is Not Enough: Bounding Knowledge in {LLM}-Based User Simulators},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  publisher = {Association for Computational Linguistics},
  year =      {2026}
}
```

## Contact

Should you have any questions, please contact Krisztian Balog at `krisztian.balog`[AT]uis.no (with [AT] replaced by @).
