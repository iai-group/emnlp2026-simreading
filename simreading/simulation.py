import copy
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Optional

from simreading.llm_connector import LLMConnector
from tqdm import tqdm

from simreading.questions import Question, Questions, QuestionType
from simreading.responses import Response, Responses

_CHECKPOINT_INTERVAL = 10

_PROMPT_PREAMBLE = (
    "You are simulating a 5th-grade student in Norway taking a "
    "reading comprehension test. Read the following text and "
    "answer the question as a typical 10-11 year old student "
    "would. The student may not always answer correctly.\n\n"
    "<TEXT>\n{text_content}\n</TEXT>\n\n"
    "<QUESTION>\n{question_text}\n</QUESTION>\n\n"
)

# Fully Norwegian (Bokmål) variant of the baseline prompt, used to
# test whether English structural instructions affect behavior.
_PROMPT_PREAMBLE_NO = (
    "Du simulerer en elev på 5. trinn i Norge som tar en "
    "leseforståelsesprøve. Les teksten nedenfor og svar på "
    "spørsmålet slik en typisk 10–11 år gammel elev ville gjort. "
    "Eleven svarer ikke alltid riktig.\n\n"
    "<TEXT>\n{text_content}\n</TEXT>\n\n"
    "<QUESTION>\n{question_text}\n</QUESTION>\n\n"
)

_PROMPT_TRUE_OR_FALSE_NO = (
    "Dette er et sant/usant-spørsmål. Avgjør om følgende påstand "
    "er sann eller usann basert på teksten.\n\n"
    "Påstand: {statement}\n\n"
    "Svar KUN med et JSON-objekt på følgende format, uten annen "
    "tekst:\n"
    '{{"answer": true}}\n'
    "eller\n"
    '{{"answer": false}}\n'
)

_PROMPT_MULTI_CHOICE_NO = (
    "Dette er et flervalgsspørsmål. Velg nøyaktig ETT svar blant "
    "alternativene nedenfor.\n\n"
    "Alternativer:\n{options_text}\n\n"
    "Svar KUN med et JSON-objekt på følgende format, uten annen "
    "tekst:\n"
    '{{"answer": "<bokstav>"}}\n'
    "der <bokstav> er bokstaven (A, B, C, ...) for alternativet "
    "du velger.\n"
)

_PROMPT_CHECKBOXES_NO = (
    "Dette er et avkryssingsspørsmål. For hvert alternativ "
    "nedenfor, avgjør om det er riktig eller ikke.\n\n"
    "Alternativer:\n{options_text}\n\n"
    "Svar KUN med et JSON-objekt på følgende format, uten annen "
    "tekst:\n"
    '{{"answer": [true, false, ...]}}\n'
    "Listen må inneholde én true/false-verdi for hvert "
    "alternativ, i samme rekkefølge som listet ovenfor.\n"
)

_PROMPT_PREAMBLE_LOWPERSONA = (
    "You are simulating a below-average 5th-grade student in Norway "
    "taking a reading comprehension test. This student finds reading "
    "difficult and answers only about two out of every three questions "
    "correctly (roughly 65-70% accuracy). Read the following text and "
    "answer the question as this struggling 10-11 year old would, "
    "making realistic mistakes on the harder questions.\n\n"
    "<TEXT>\n{text_content}\n</TEXT>\n\n"
    "<QUESTION>\n{question_text}\n</QUESTION>\n\n"
)

_PROMPT_EXTRACT_PROPOSITIONS_TS = (
    "You are simulating a 5th-grade student in Norway taking a "
    "reading comprehension test. You have just read the question "
    "below and are now scanning the text specifically to find the answer. "
    "As a young reader, you cannot process the whole text at once — "
    "you only focus on a few key details that seem to match the question.\n\n"
    "Scan the text and extract at most {capacity} distinct propositions "
    "(paraphrased atomic facts from the text) that share keywords or "
    "seem most directly related to what the question is asking. "
    "Do NOT answer the question yet.\n\n"
    "<TEXT>\n{text_content}\n</TEXT>\n\n"
    "<QUESTION>\n{question_text}\n</QUESTION>\n\n"
    "Respond with ONLY a JSON object in the following format, "
    "with no other text:\n"
    '{{"propositions": ["proposition 1", "proposition 2", ...]}}\n'
)

_PROMPT_EXTRACT_PROPOSITIONS_SPR = (
    "You are simulating a 5th-grade student in Norway taking a "
    "reading comprehension test. You are about to read a text, "
    "and you'll be asked questions about this text. As a young "
    "reader, you cannot remember everything from the text — "
    "only a few key facts.\n\n"
    "Read the text and identify at most {capacity} distinct "
    "propositions (paraphrased atomic facts from the text) that "
    "you think are most relevant to the questions you'll be asked. "
    "Do not answer any questions yet.\n\n"
    "<TEXT>\n{text_content}\n</TEXT>\n\n"
    "Respond with ONLY a JSON object in the following format, "
    "with no other text:\n"
    '{{"propositions": ["proposition 1", "proposition 2", ...]}}\n'
)

_PROMPT_PREAMBLE_CBUS = (
    "You are simulating a 5th-grade student in Norway taking a "
    "reading comprehension test. You no longer have access to "
    "the full text — you can only rely on the propositions you "
    "remembered from reading. Answer the question as a typical "
    "10-11 year old student would, based ONLY on the "
    "propositions below. The student may not always answer "
    "correctly.\n\n"
    "<PROPOSITIONS>\n{propositions}\n</PROPOSITIONS>\n\n"
    "<QUESTION>\n{question_text}\n</QUESTION>\n\n"
)

_PROMPT_TRUE_OR_FALSE = (
    "This is a true or false question. Decide whether the "
    "following statement is true or false based on the "
    "text.\n\n"
    "Statement: {statement}\n\n"
    "Respond with ONLY a JSON object in the following "
    "format, with no other text:\n"
    '{{"answer": true}}\n'
    "or\n"
    '{{"answer": false}}\n'
)

_PROMPT_MULTI_CHOICE = (
    "This is a multiple choice question. Choose exactly ONE "
    "answer from the options below.\n\n"
    "Options:\n{options_text}\n\n"
    "Respond with ONLY a JSON object in the following "
    "format, with no other text:\n"
    '{{"answer": "<letter>"}}\n'
    "where <letter> is the letter (A, B, C, ...) of your "
    "chosen option.\n"
)

_PROMPT_CHECKBOXES = (
    "This is a checkbox question. For each option below, "
    "decide whether it is correct or not.\n\n"
    "Options:\n{options_text}\n\n"
    "Respond with ONLY a JSON object in the following "
    "format, with no other text:\n"
    '{{"answer": [true, false, ...]}}\n'
    "The list must contain one true/false value for each "
    "option, in the same order as listed above.\n"
)


# Inner-monologue variants. Each adds a "thought" key to the JSON
# response containing a brief Tanke-logg in Norwegian Bokmål, written
# in the voice of a 10-year-old reasoning from {source} (e.g. "the
# text" for the baseline LLM, "the Remembered Facts" for CBUS).

_PROMPT_TRUE_OR_FALSE_WITH_THOUGHT = (
    "This is a true or false question. Decide whether the "
    "following statement is true or false based on the "
    "text.\n\n"
    "Statement: {statement}\n\n"
    "First, write a brief 'Tanke-logg' (inner monologue) in "
    "Norwegian Bokmål showing your 10-year-old reasoning "
    "process based only on {source}. Then state your final "
    "choice.\n\n"
    "Respond with ONLY a JSON object in the following "
    "format, with no other text:\n"
    '{{"thought": "<your tanke-logg in bokmål>", "answer": true}}\n'
    "or\n"
    '{{"thought": "<your tanke-logg in bokmål>", "answer": false}}\n'
)

_PROMPT_MULTI_CHOICE_WITH_THOUGHT = (
    "This is a multiple choice question. Choose exactly ONE "
    "answer from the options below.\n\n"
    "Options:\n{options_text}\n\n"
    "First, write a brief 'Tanke-logg' (inner monologue) in "
    "Norwegian Bokmål showing your 10-year-old reasoning "
    "process based only on {source}. Then state your final "
    "choice.\n\n"
    "Respond with ONLY a JSON object in the following "
    "format, with no other text:\n"
    '{{"thought": "<your tanke-logg in bokmål>", "answer": "<letter>"}}\n'
    "where <letter> is the letter (A, B, C, ...) of your "
    "chosen option.\n"
)

_PROMPT_CHECKBOXES_WITH_THOUGHT = (
    "This is a checkbox question. For each option below, "
    "decide whether it is correct or not.\n\n"
    "Options:\n{options_text}\n\n"
    "First, write a brief 'Tanke-logg' (inner monologue) in "
    "Norwegian Bokmål showing your 10-year-old reasoning "
    "process based only on {source}. Then state your final "
    "choice.\n\n"
    "Respond with ONLY a JSON object in the following "
    "format, with no other text:\n"
    '{{"thought": "<your tanke-logg in bokmål>", '
    '"answer": [true, false, ...]}}\n'
    "The list must contain one true/false value for each "
    "option, in the same order as listed above.\n"
)


class Simulator(str, Enum):
    """Available simulation strategies."""

    RANDOM = "random"
    LLM_BASELINE = "llm_baseline"
    LLM_BASELINE_NO = "llm_baseline_no"
    LLM_INNER_MONOLOGUE = "llm_innermonologue"
    LLM_LOWPERSONA = "llm_lowpersona"
    CBUS_TS = "cbus_ts"
    CBUS_SPR = "cbus_spr"
    CBUS_SPR_DROPOUT = "cbus_spr_dropout"


_DEFAULT_CAPACITY = 3


class Simulation:
    """Simulates student responses by predicting answers for each response.

    Takes a set of real responses and produces a new set where the
    student answers are replaced with predicted (simulated) values.
    The ground-truth fields are preserved so that correctness can be
    evaluated after simulation.

    Attributes:
        questions: Questions database used to look up texts, questions,
            and answer options.
        simulator: The simulation strategy to use.
    """

    def __init__(
        self,
        questions: Questions,
        simulator: Simulator = Simulator.RANDOM,
        verbose: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        capacity: int = _DEFAULT_CAPACITY,
        concurrency: int = 1,
    ) -> None:
        """Initialises a Simulation instance.

        Args:
            questions: Questions database for looking up question
                metadata (texts, questions, answer options).
            simulator: The simulation strategy. Defaults to
                ``Simulator.RANDOM``.
            verbose: If ``True``, prints prompts and LLM responses
                to stdout.
            provider: LLM provider name (e.g. ``"ollama"``,
                ``"openrouter"``).  Defaults to the value in
                ``llm_config.yaml``.
            model: Model name override.  Defaults to the provider's
                ``default_model`` in ``llm_config.yaml``.
            capacity: Working-memory capacity (max propositions) for
                the CBUS-TS and CBUS-SPR simulators. Ignored by
                other simulators.
            concurrency: Maximum number of students processed
                concurrently (i.e. the cap on simultaneous LLM calls).
                Each student is answered independently with its own
                per-student cache, so responses are freshly sampled
                per student while a single reading (SPR Stage 1) is
                still reused across that student's questions on a text.
        """
        self.questions = questions
        self.simulator = simulator
        self.verbose = verbose
        self.capacity = capacity
        self.concurrency = max(1, int(concurrency))
        self._llm: Optional[LLMConnector] = None

        if self.simulator != Simulator.RANDOM:
            self._llm = LLMConnector(provider=provider, model=model)

    def _llm_generate(self, prompt: str) -> str:
        """Sends a prompt to the LLM and returns the raw response text.

        Delegates to the ``LLMConnector`` which handles provider
        selection, model routing, and generation parameters.

        Args:
            prompt: The prompt string to send.

        Returns:
            The raw response text from the LLM.
        """
        return self._llm.generate(prompt)

    def simulate(
        self,
        responses: Responses,
        output_file: Optional[str] = None,
        resume_from: Optional[Responses] = None,
    ) -> Responses:
        """Simulates answers for all responses in the collection.

        Creates a deep copy of the responses, removes all student
        answers to prevent data leakage, then generates predicted
        answers and evaluates correctness. Progress is shown via a
        progress bar over students.

        Args:
            responses: The original set of responses to simulate.
            output_file: Optional path to save checkpoint results
                every ``_CHECKPOINT_INTERVAL`` students.
            resume_from: Optional pre-existing partial result. Any
                student whose responses are all already predicted in
                ``resume_from`` is skipped; their predictions are
                copied verbatim into the output.

        Returns:
            A new Responses collection with simulated answers.
        """
        completed_ids: set[str] = set()
        if resume_from is not None:
            input_ids = set(responses._by_student.keys())
            for sid in input_ids & set(resume_from._by_student.keys()):
                resps = resume_from.get_by_student(sid)
                if resps and all(r.correct is not None for r in resps):
                    completed_ids.add(sid)

            completed_resps: list[Response] = []
            for sid in completed_ids:
                completed_resps.extend(resume_from.get_by_student(sid))

            redo_resps = [
                copy.deepcopy(r)
                for r in responses._responses
                if r.student_id not in completed_ids
            ]
            redo_collection = Responses(redo_resps)
            redo_collection.remove_student_answers()

            simulated = Responses(completed_resps + redo_collection._responses)
        else:
            simulated = Responses(copy.deepcopy(responses._responses))
            simulated.remove_student_answers()

        todo = [
            sid for sid in simulated._by_student.keys()
            if sid not in completed_ids
        ]

        done = 0
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(self._predict_student, simulated, sid): sid
                for sid in todo
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Simulating students",
                unit="student",
            ):
                future.result()  # propagate any worker exception
                done += 1
                if output_file and done % _CHECKPOINT_INTERVAL == 0:
                    simulated.to_json(output_file)

        return simulated

    def _predict_student(self, simulated: Responses, student_id: str) -> None:
        """Predicts all of one student's responses (in place).

        Uses a fresh per-student cache so answers are sampled
        independently for each student, while a single reading (SPR
        Stage 1) and multi-option checkbox calls are still reused
        across that student's own responses. Safe to run concurrently
        across students: the cache is local and each student's Response
        objects are distinct.
        """
        cache: dict = {}
        for response in simulated.get_by_student(student_id):
            self.predict(response, cache)

    def predict(self, response: Response, cache: Optional[dict] = None) -> None:
        """Predicts and populates the student's answer on a Response.

        Assumes student answer fields have already been cleared
        (e.g. via ``Responses.remove_student_answers``). Generates a
        predicted answer, populates the corresponding fields, and
        updates ``response.correct`` by calling
        ``response.is_student_correct()``.

        Each Response carries both ``sanity_text_id`` and
        ``question_id``, so a question that appears under multiple
        texts produces a distinct Response per text and each is
        predicted independently with the correct text context.

        Args:
            response: The Response object to populate in place.
            cache: A per-student dict used to reuse LLM calls within a
                single student (SPR Stage 1 across a text's questions,
                and a single call for a checkbox question's options).
                Pass the same dict for all of one student's responses;
                omit (a fresh dict is created) to answer this response
                in isolation.

        Raises:
            ValueError: If the question type is not recognised.
        """
        if cache is None:
            cache = {}
        if self.simulator == Simulator.RANDOM:
            self._predict_random(response)
        elif self.simulator == Simulator.CBUS_TS:
            self._predict_cbus_ts(response, cache)
        elif self.simulator == Simulator.CBUS_SPR:
            self._predict_cbus_spr(response, cache)
        elif self.simulator == Simulator.CBUS_SPR_DROPOUT:
            self._predict_cbus_spr(response, cache, dropout=True)
        else:
            self._predict_llm(response, cache)

        response.is_student_correct()

    # ------------------------------------------------------------------
    # Random predictor
    # ------------------------------------------------------------------

    def _predict_random(self, response: Response) -> None:
        """Fills a response with a random prediction.

        Args:
            response: The Response object to populate in place.

        Raises:
            ValueError: If the question type is not recognised.
        """
        if response.type == QuestionType.TRUE_OR_FALSE:
            response.student_answer = random.choice([True, False])

        elif response.type == QuestionType.CHECKBOXES:
            if response.options:
                for opt in response.options:
                    opt.student_answer = random.choice([True, False])

        elif response.type == QuestionType.MULTI_CHOICE:
            question_data = self.questions.get_question(response.question_id)
            if question_data and hasattr(question_data, "options"):
                valid_options = [opt.sanity_answer_key for opt in question_data.options]
                if valid_options:
                    response.selected_sanity_option_keys = [
                        random.choice(valid_options)
                    ]
        else:
            raise ValueError(f"Unknown question type: {response.type}")

    # ------------------------------------------------------------------
    # LLM-based predictor
    # ------------------------------------------------------------------

    def create_prompt(self, text_id: str, question_id: str) -> str:
        """Creates an LLM prompt for predicting a student's response.

        Builds a prompt that instructs the LLM to simulate a Norwegian
        5th-grader answering a reading comprehension question. The
        prompt format and expected JSON output vary by question type.

        Args:
            text_id: The ``sanity_text_id`` of the reading passage.
            question_id: The ``sanity_question_key`` of the question.

        Returns:
            The formatted prompt string.

        Raises:
            ValueError: If the text or question is not found, or the
                question type is not recognised.
        """
        text_data = self.questions.get_text(text_id)
        question_data = self.questions.get_question(question_id)

        if not text_data:
            raise ValueError(f"Text not found: {text_id}")
        if not question_data:
            raise ValueError(f"Question not found: {question_id}")

        text_content = text_data.text
        question_text = question_data.question

        norwegian = self.simulator == Simulator.LLM_BASELINE_NO
        if norwegian:
            template = _PROMPT_PREAMBLE_NO
        elif self.simulator == Simulator.LLM_LOWPERSONA:
            template = _PROMPT_PREAMBLE_LOWPERSONA
        else:
            template = _PROMPT_PREAMBLE
        preamble = template.format(
            text_content=text_content,
            question_text=question_text,
        )

        with_thought = self.simulator == Simulator.LLM_INNER_MONOLOGUE
        instructions = self._build_instructions(
            question_data,
            with_thought=with_thought,
            source="the text",
            norwegian=norwegian,
        )

        return preamble + instructions

    def _build_instructions(
        self,
        question_data: Question,
        with_thought: bool = False,
        source: str = "the text",
        norwegian: bool = False,
    ) -> str:
        """Builds type-specific answer instructions for an LLM prompt.

        Args:
            question_data: The question object.
            with_thought: If True, use the inner-monologue prompt
                variant that asks the LLM to emit a ``"thought"``
                field alongside ``"answer"``.
            source: Phrasing inserted into the thought instruction
                (e.g. ``"the text"`` for the baseline LLM, ``"the
                Remembered Facts"`` for CBUS). Ignored when
                ``with_thought`` is False.
            norwegian: If True, use the fully Norwegian (Bokmål)
                instruction templates. Mutually exclusive with
                ``with_thought``.

        Returns:
            The formatted instruction string for the question type.

        Raises:
            ValueError: If the question type is not recognised.
        """
        q_type = QuestionType(question_data.type)

        if q_type == QuestionType.TRUE_OR_FALSE:
            options = getattr(question_data, "options", [])
            statement = options[0].option if options else ""
            if norwegian:
                tpl = _PROMPT_TRUE_OR_FALSE_NO
            elif with_thought:
                tpl = _PROMPT_TRUE_OR_FALSE_WITH_THOUGHT
            else:
                tpl = _PROMPT_TRUE_OR_FALSE
            return tpl.format(statement=statement, source=source)

        elif q_type in (QuestionType.MULTI_CHOICE, QuestionType.CHECKBOXES):
            options = getattr(question_data, "options", [])
            options_text = "\n".join(
                f"  {chr(65 + i)}) {opt.option}" for i, opt in enumerate(options)
            )
            if q_type == QuestionType.MULTI_CHOICE:
                if norwegian:
                    tpl = _PROMPT_MULTI_CHOICE_NO
                elif with_thought:
                    tpl = _PROMPT_MULTI_CHOICE_WITH_THOUGHT
                else:
                    tpl = _PROMPT_MULTI_CHOICE
            else:
                if norwegian:
                    tpl = _PROMPT_CHECKBOXES_NO
                elif with_thought:
                    tpl = _PROMPT_CHECKBOXES_WITH_THOUGHT
                else:
                    tpl = _PROMPT_CHECKBOXES
            return tpl.format(options_text=options_text, source=source)

        else:
            raise ValueError(f"Unknown question type: {q_type}")

    def _call_llm(self, text_id: str, question_id: str, cache: dict) -> dict:
        """Calls the LLM and returns the parsed response.

        Results are stored in the per-student ``cache`` by
        ``(text_id, question_id)`` so that checkbox questions (which
        have multiple Response objects per question) only trigger a
        single LLM call for that student.

        Args:
            text_id: The ``sanity_text_id`` of the reading passage.
            question_id: The ``sanity_question_key`` of the question.
            cache: Per-student cache dict.

        Returns:
            The parsed JSON response as a dictionary.
        """
        key = (text_id, question_id)
        if key in cache:
            return cache[key]

        prompt = self.create_prompt(text_id, question_id)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"PROMPT [{text_id} / {question_id}]")
            print(f"{'='*60}")
            print(prompt)

        raw_text = self._llm_generate(prompt)
        parsed = self._parse_llm_response(raw_text)

        if self.verbose:
            print(f"\nRESPONSE: {raw_text}")
            print(f"PARSED:   {parsed}")

        cache[key] = parsed
        return parsed

    def _predict_llm(self, response: Response, cache: dict) -> None:
        """Fills a response using an LLM prediction.

        Calls the LLM (or uses a stored result), then maps the
        human-friendly answer format back to internal fields:

        - **trueOrFalse**: ``{"answer": true/false}``
        - **multiChoice**: ``{"answer": "B"}`` → mapped to
          ``sanity_answer_key`` using option index
        - **checkboxes**: ``{"answer": [true, false, ...]}`` → each
          Response's ``student_answer`` is looked up by its position

        Args:
            response: The Response object to populate in place.

        Raises:
            ValueError: If the question type is not recognised.
        """
        parsed_answer = self._call_llm(
            response.sanity_text_id, response.question_id, cache,
        )
        self._apply_parsed_answer(response, parsed_answer)

    # ------------------------------------------------------------------
    # CBUS-TS predictor (Targeted Scanning: per-question extraction)
    # ------------------------------------------------------------------

    def _extract_propositions_ts(
        self,
        text_id: str,
        question_id: str,
        cache: dict,
    ) -> list[str]:
        """Stage 1 for CBUS-TS: scan the text given the question and
        extract up to ``capacity`` propositions.

        Prompts the LLM (in the student persona) to extract at most
        ``self.capacity`` paraphrased atomic facts from the text that
        seem most directly related to the question. The result is
        cached by ``(text_id, question_id)``.

        Args:
            text_id: The ``sanity_text_id`` of the reading passage.
            question_id: The ``question_id`` of the question.

        Returns:
            A list of proposition strings (length up to ``capacity``).
        """
        key = ("cbus_ts_extract", text_id, question_id)
        if key in cache:
            return cache[key]

        text_data = self.questions.get_text(text_id)
        question_data = self.questions.get_question(question_id)
        text_content = text_data.text if text_data else ""
        question_text = question_data.question if question_data else ""

        prompt = _PROMPT_EXTRACT_PROPOSITIONS_TS.format(
            capacity=self.capacity,
            text_content=text_content,
            question_text=question_text,
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"CBUS-TS EXTRACT [{text_id} / {question_id}]")
            print(f"{'='*60}")
            print(prompt)

        raw_text = self._llm_generate(prompt)
        parsed = self._parse_llm_response(raw_text)
        propositions = parsed.get("propositions", [])
        if not isinstance(propositions, list):
            propositions = []
        propositions = [str(p) for p in propositions[: self.capacity]]

        if self.verbose:
            print(f"\nRESPONSE: {raw_text}")
            print(f"EXTRACTED PROPOSITIONS: {propositions}")

        cache[key] = propositions
        return propositions

    def _predict_cbus_ts(self, response: Response, cache: dict) -> None:
        """Fills a response using the CBUS-TS (Targeted Scanning)
        pipeline.

        Stage 1: scan the text *given the question* and extract up to
        ``capacity`` propositions that seem related to the question.
        Stage 2: answer the question using only those propositions,
        with the source text purged from the prompt.

        Both stages cache by ``(text_id, question_id)`` so multiple
        Response objects per question (e.g. checkboxes) only trigger
        one extraction and one answer call.

        Args:
            response: The Response object to populate in place.

        Raises:
            ValueError: If the question type is not recognised.
        """
        text_id = response.sanity_text_id
        question_id = response.question_id

        propositions = self._extract_propositions_ts(text_id, question_id, cache)
        response.propositions = list(propositions)

        answer_key = ("cbus_ts_answer", text_id, question_id)
        if answer_key in cache:
            parsed_answer = cache[answer_key]
        else:
            question_data = self.questions.get_question(question_id)
            question_text = question_data.question if question_data else ""
            propositions_text = (
                "\n".join(f"- {p}" for p in propositions)
                if propositions
                else "- (no propositions remembered)"
            )

            preamble = _PROMPT_PREAMBLE_CBUS.format(
                propositions=propositions_text,
                question_text=question_text,
            )
            instructions = self._build_instructions(
                question_data,
                with_thought=True,
                source="the Remembered Facts",
            )
            prompt = preamble + instructions

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"CBUS-TS ANSWER [{text_id} / {question_id}]")
                print(f"{'='*60}")
                print(prompt)

            raw_response = self._llm_generate(prompt)
            parsed_answer = self._parse_llm_response(raw_response)

            if self.verbose:
                print(f"\nRESPONSE: {raw_response}")
                print(f"PARSED:   {parsed_answer}")

            cache[answer_key] = parsed_answer

        self._apply_parsed_answer(response, parsed_answer)

    # ------------------------------------------------------------------
    # CBUS-SPR predictor (Single-Pass Reading: one extraction per text)
    # ------------------------------------------------------------------

    def _extract_propositions_spr(
        self, text_id: str, cache: dict, dropout: bool = False,
    ) -> list[str]:
        """Stage 1 for CBUS-SPR: extract up to ``capacity`` text-level
        propositions. The student does not yet know the questions.

        Cached by ``text_id`` alone so the same propositions are reused
        across every question on the text.

        When ``dropout`` is True, the model instead extracts a larger
        salient pool and a random subset of ``capacity`` propositions is
        retained. This ablates the LLM's salience-based *selection* while
        holding the capacity bound fixed: the amount remembered is the
        same, but which propositions survive is random rather than
        most-salient-first.

        Args:
            text_id: The ``sanity_text_id`` of the reading passage.
            dropout: If True, use the random proposition-dropout variant.

        Returns:
            A list of proposition strings (length up to ``capacity``).
        """
        key = (
            "cbus_spr_dropout_extract" if dropout else "cbus_spr_extract",
            text_id,
        )
        if key in cache:
            return cache[key]

        text_data = self.questions.get_text(text_id)
        text_content = text_data.text if text_data else ""

        # For dropout, extract a generous salient pool, then randomly
        # retain ``capacity`` of them below.
        extract_cap = max(12, 3 * self.capacity) if dropout else self.capacity
        prompt = _PROMPT_EXTRACT_PROPOSITIONS_SPR.format(
            capacity=extract_cap,
            text_content=text_content,
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"CBUS-SPR EXTRACT [{text_id}]")
            print(f"{'='*60}")
            print(prompt)

        raw_text = self._llm_generate(prompt)
        parsed = self._parse_llm_response(raw_text)
        propositions = parsed.get("propositions", [])
        if not isinstance(propositions, list):
            propositions = []
        propositions = [str(p) for p in propositions]
        if dropout:
            # Randomly retain ``capacity`` of the extracted salient pool.
            if len(propositions) > self.capacity:
                propositions = random.sample(propositions, self.capacity)
        else:
            propositions = propositions[: self.capacity]

        if self.verbose:
            print(f"\nRESPONSE: {raw_text}")
            print(f"EXTRACTED PROPOSITIONS: {propositions}")

        cache[key] = propositions
        return propositions

    def _predict_cbus_spr(
        self, response: Response, cache: dict, dropout: bool = False,
    ) -> None:
        """Fills a response using the CBUS-SPR (Single-Pass Reading)
        pipeline.

        Stage 1 fires once per text (without seeing any question) to
        extract ``capacity`` propositions. Stage 2 answers the
        question from those propositions, just like CBUS-TS.

        Args:
            response: The Response object to populate in place.
            dropout: If True, use the random proposition-dropout
                variant in Stage 1 (see ``_extract_propositions_spr``).

        Raises:
            ValueError: If the question type is not recognised.
        """
        text_id = response.sanity_text_id
        question_id = response.question_id

        propositions = self._extract_propositions_spr(
            text_id, cache, dropout=dropout,
        )
        response.propositions = list(propositions)

        answer_prefix = (
            "cbus_spr_dropout_answer" if dropout else "cbus_spr_answer"
        )
        answer_key = (answer_prefix, text_id, question_id)
        if answer_key in cache:
            parsed_answer = cache[answer_key]
        else:
            question_data = self.questions.get_question(question_id)
            question_text = question_data.question if question_data else ""
            propositions_text = (
                "\n".join(f"- {p}" for p in propositions)
                if propositions
                else "- (no propositions remembered)"
            )

            preamble = _PROMPT_PREAMBLE_CBUS.format(
                propositions=propositions_text,
                question_text=question_text,
            )
            instructions = self._build_instructions(
                question_data,
                with_thought=True,
                source="the Remembered Facts",
            )
            prompt = preamble + instructions

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"CBUS-SPR ANSWER [{text_id} / {question_id}]")
                print(f"{'='*60}")
                print(prompt)

            raw_response = self._llm_generate(prompt)
            parsed_answer = self._parse_llm_response(raw_response)

            if self.verbose:
                print(f"\nRESPONSE: {raw_response}")
                print(f"PARSED:   {parsed_answer}")

            cache[answer_key] = parsed_answer

        self._apply_parsed_answer(response, parsed_answer)

    def _apply_parsed_answer(
        self,
        response: Response,
        parsed: dict,
    ) -> None:
        """Maps a parsed LLM answer onto a Response object.

        Interprets the parsed JSON according to the response's
        question type and populates the appropriate fields:

        - **trueOrFalse**: ``{"answer": true/false}``
        - **multiChoice**: ``{"answer": "B"}`` → mapped to
          ``sanity_answer_key`` using option index
        - **checkboxes**: ``{"answer": [true, false, ...]}`` → each
          Response's ``student_answer`` is looked up by its position

        Args:
            response: The Response object to populate in place.
            parsed: The parsed JSON dict from the LLM.

        Raises:
            ValueError: If the question type is not recognised.
        """
        question_data = self.questions.get_question(response.question_id)
        options = getattr(question_data, "options", []) if question_data else []

        if response.type == QuestionType.TRUE_OR_FALSE:
            answer = parsed.get("answer")
            if isinstance(answer, bool):
                response.student_answer = answer
            else:
                response.student_answer = str(answer).strip().lower() == "true"

        elif response.type == QuestionType.MULTI_CHOICE:
            letter = str(parsed.get("answer", "")).strip().upper()
            is_valid = len(letter) == 1 and letter.isalpha()
            idx = ord(letter) - ord("A") if is_valid else -1
            chosen = (
                options[idx]
                if 0 <= idx < len(options)
                else (options[0] if options else None)
            )
            if chosen:
                response.selected_sanity_option_keys = [chosen.sanity_answer_key]

        elif response.type == QuestionType.CHECKBOXES:
            answers = parsed.get("answer", [])
            if response.options:
                for i, opt in enumerate(response.options):
                    opt.student_answer = (
                        bool(answers[i])
                        if isinstance(answers, list) and i < len(answers)
                        else False
                    )

        else:
            raise ValueError(f"Unknown question type: {response.type}")

    @staticmethod
    def _parse_llm_response(raw_text: str) -> dict:
        """Parses a JSON response from the LLM.

        Attempts to extract a JSON object from the LLM's raw text
        output, handling cases where the JSON may be wrapped in
        markdown code fences or surrounded by extra text.

        Args:
            raw_text: The raw text output from the LLM.

        Returns:
            The parsed JSON as a dictionary, or an empty dict if
            parsing fails.
        """
        # Strip markdown code fences if present
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        return {}
