"""Runs a new simulator over the item bank and writes question-stats.

The public release ships no per-student exposure, so a simulation cannot
mirror individual students. Instead this builds *synthetic readers*: for
each text, ``n_readers`` independent readers (from the shipped per-text
reader counts) each read that one text and answer all its questions. This
reuses the exact CBUS/baseline machinery (``Simulation.simulate``) — CBUS-SPR
still encodes once per reader — and reproduces the item-centric statistics up
to sampling noise.

The result is written as a per-run ``question_stats`` CSV under
``data/aggregates/simulation/<method>/``, which ``run_evaluation.py`` scores
for Abs. Gap, ECE, Pearson, and Spearman. It does **not** write a score
histogram: single-text synthetic readers are not comparable to real
multi-text students, so JSD / Figure 1 are unavailable for new runs (see
RELEASE_PLAN.md). To reproduce the paper's JSD, use the shipped precomputed
aggregates.

Requires an LLM: copy ``llm_config.example.yaml`` to ``llm_config.yaml`` and
set your provider/key. Use ``--limit`` to smoke-test on a few texts.

Usage:
    python scripts/run_simulation.py --simulator cbus_spr --model <id>
    python scripts/run_simulation.py --simulator random --limit 5
"""

import argparse
import csv
import os

from simreading import Questions, Response, Responses, Simulation, Simulator
from simreading.questions import QuestionType
from simreading.responses import OptionResponse
from simreading.experiments import cbus_default, method_dirname

READER_COUNTS = "data/aggregates/human/text_reader_counts_test.csv"
OUT_DIR = "data/aggregates/simulation"
LLM_CONFIG_PATH = "llm_config.yaml"

_ANSWERABLE = {
    QuestionType.TRUE_OR_FALSE, QuestionType.MULTI_CHOICE, QuestionType.CHECKBOXES,
}


def _short_model_name(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def _resolve_model(provider, model):
    if model:
        return model
    try:
        import yaml
        with open(LLM_CONFIG_PATH) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        return None
    prov = provider or config.get("default_provider")
    return config.get("providers", {}).get(prov, {}).get("default_model")


def _synthetic_response(q, text_id: str, student_id: str) -> Response | None:
    """Builds one ground-truth-only Response for a synthetic reader."""
    q_type = QuestionType(q.type)
    if q_type == QuestionType.TRUE_OR_FALSE:
        return Response(
            type=q_type, question_id=q.id, sanity_text_id=text_id,
            student_id=student_id, is_correct=q.is_correct,
        )
    if q_type == QuestionType.MULTI_CHOICE:
        correct = [o.sanity_answer_key for o in q.options if o.is_correct]
        return Response(
            type=q_type, question_id=q.id, sanity_text_id=text_id,
            student_id=student_id, correct_answer_sanity_option_keys=correct,
        )
    if q_type == QuestionType.CHECKBOXES:
        options = [
            OptionResponse(sanity_answer_key=o.sanity_answer_key,
                           is_correct=o.is_correct)
            for o in q.options
        ]
        return Response(
            type=q_type, question_id=q.id, sanity_text_id=text_id,
            student_id=student_id, options=options,
        )
    return None


def _build_synthetic(questions: Questions, reader_counts: dict, limit) -> Responses:
    texts = questions.get_all_texts()
    if limit is not None:
        texts = texts[:limit]
    records: list[Response] = []
    for text in texts:
        n = reader_counts.get(text.sanity_text_id, 0)
        answerable = [q for q in text.questions if QuestionType(q.type) in _ANSWERABLE]
        for i in range(n):
            sid = f"{text.sanity_text_id}#r{i}"
            for q in answerable:
                r = _synthetic_response(q, text.sanity_text_id, sid)
                if r is not None:
                    records.append(r)
    return Responses(records)


def _load_reader_counts() -> dict:
    counts = {}
    with open(READER_COUNTS, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["sanity_text_id"]] = int(row["n_readers"])
    return counts


def _write_question_stats(path: str, responses: Responses) -> None:
    stats: dict = {}
    for r in responses:
        cell = stats.setdefault((r.sanity_text_id, r.question_id), [0, 0])
        if r.correct is True:
            cell[0] += 1
        elif r.correct is False:
            cell[1] += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sanity_text_id", "question_id", "n_correct", "n_incorrect"])
        for (t, q), (nc, ni) in sorted(stats.items()):
            w.writerow([t, q, nc, ni])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulator", default="random",
                    choices=[s.value for s in Simulator])
    ap.add_argument("--model", default=None)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--capacity", type=int, default=None)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only simulate the first N texts (smoke test).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.simulator != Simulator.RANDOM.value:
        args.model = _resolve_model(args.provider, args.model)
        if not args.model:
            raise SystemExit(
                "ERROR: could not determine model. Pass --model or set a "
                f"default in {LLM_CONFIG_PATH}."
            )
    if args.capacity is None:
        if args.simulator == Simulator.CBUS_TS.value:
            args.capacity = cbus_default("ts")
        elif args.simulator in (Simulator.CBUS_SPR.value,
                                Simulator.CBUS_SPR_DROPOUT.value):
            args.capacity = cbus_default("spr")

    questions = Questions()
    reader_counts = _load_reader_counts()
    synthetic = _build_synthetic(questions, reader_counts, args.limit)
    print(f"Built {len(synthetic)} synthetic responses "
          f"({len(synthetic._by_student)} readers).")

    model_short = _short_model_name(args.model) if args.model else None
    method = method_dirname(args.simulator, model_short, args.capacity)
    out_path = os.path.join(OUT_DIR, method, f"r{args.run}_question_stats.csv")

    simulation = Simulation(
        questions, simulator=Simulator(args.simulator), verbose=args.verbose,
        provider=args.provider, model=args.model,
        capacity=args.capacity or 3, concurrency=args.concurrency,
    )
    simulated = simulation.simulate(synthetic)

    _write_question_stats(out_path, simulated)
    n_correct = sum(1 for r in simulated if r.correct is True)
    n_done = sum(1 for r in simulated if r.correct is not None)
    print(f"Accuracy: {n_correct / n_done:.3f}  ({n_done} answered)")
    print(f"Wrote {out_path}")
    print("Note: no score histogram written (JSD unavailable for new runs).")


if __name__ == "__main__":
    main()
