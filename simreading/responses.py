"""Response dataclasses and response-loading utilities.

Provides typed representations for student responses to reading
comprehension questions, a ``Responses`` collection class with
indexing and statistics, and JSON serialization/deserialization.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from simreading.questions import QuestionType


@dataclass
class OptionResponse:
    """A student's response to a single answer option.

    Used for checkboxes questions to record per-option student answers
    alongside the ground truth.

    Attributes:
        sanity_answer_key: Sanity CMS key for this answer option.
        is_correct: Whether this option should be selected (ground
            truth).
        student_answer: Whether the student selected this option.
            ``None`` when the student answer has been cleared for
            simulation.
    """

    sanity_answer_key: str
    is_correct: bool
    student_answer: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptionResponse:
        """Creates an OptionResponse from a dictionary."""
        return cls(
            sanity_answer_key=data['sanity_answer_key'],
            is_correct=data['is_correct'],
            student_answer=data.get('student_answer'),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialises this option response to a dictionary."""
        return {
            'sanity_answer_key': self.sanity_answer_key,
            'student_answer': self.student_answer,
            'is_correct': self.is_correct,
        }


@dataclass
class Response:
    """A single student response to a question.

    Each Response maps 1:1 to a ``question_id`` matching a
    :class:`Question` instance.  For checkboxes questions the
    per-option data is stored in :attr:`options`.

    Attributes:
        type: The question type (trueOrFalse, checkboxes, or
            multiChoice).
        question_id: Identifier matching ``Question.id``.
        sanity_text_id: Identifier of the text this response belongs to.
        student_id: Identifier of the student who gave this response.
        correct: Whether the student's answer is correct overall.
        student_answer: The student's true/false answer
            (trueOrFalse only).
        is_correct: Ground truth for the statement (trueOrFalse only).
        options: Per-option response data (checkboxes only).
        selected_sanity_option_key: Option key selected by the student
            (multiChoice only).
        correct_answer_sanity_option_key: Ground truth option key
            (multiChoice only).
    """

    type: QuestionType
    question_id: str
    sanity_text_id: str
    student_id: str
    correct: bool | None = None

    # TrueOrFalse
    student_answer: bool | None = None
    is_correct: bool | None = None

    # Checkboxes
    options: list[OptionResponse] | None = None

    # MultiChoice
    selected_sanity_option_keys: list[str] | None = None
    correct_answer_sanity_option_keys: list[str] | None = None

    # CBUS: the Stage-1 propositions (episodic buffer contents) the answer
    # was produced from. Set only by the CBUS simulators; None otherwise.
    propositions: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Response:
        """Creates a Response from a dictionary.

        Args:
            data: Dictionary containing response fields.  Must include
                a valid ``type`` field matching a ``QuestionType`` value.

        Returns:
            A new Response instance.

        Raises:
            ValueError: If ``type`` is missing or not a recognised
                question type.
        """
        q_type = QuestionType(data['type'])

        options = None
        if 'options' in data:
            options = [OptionResponse.from_dict(o) for o in data['options']]

        return cls(
            type=q_type,
            question_id=data.get('question_id', ''),
            sanity_text_id=data.get('sanity_text_id', ''),
            student_id=data.get('student_id', ''),
            correct=data.get('correct'),
            student_answer=data.get('student_answer'),
            is_correct=data.get('is_correct'),
            options=options,
            selected_sanity_option_keys=data.get('selected_sanity_option_keys'),
            correct_answer_sanity_option_keys=data.get(
                'correct_answer_sanity_option_keys'
            ),
            propositions=data.get('propositions'),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialises this response to a dictionary.

        Returns:
            A dictionary containing the response fields appropriate for
            its question type.  Structural fields (``question_id``,
            ``sanity_text_id``, ``student_id``) are omitted as they are
            represented by the nesting structure in the JSON file.
        """
        result: dict[str, Any] = {
            'type': self.type.value,
            'correct': self.correct,
        }

        if self.type == QuestionType.TRUE_OR_FALSE:
            result['student_answer'] = self.student_answer
            result['is_correct'] = self.is_correct
        elif self.type == QuestionType.CHECKBOXES:
            if self.options is not None:
                result['options'] = [o.to_dict() for o in self.options]
        elif self.type == QuestionType.MULTI_CHOICE:
            result['selected_sanity_option_keys'] = self.selected_sanity_option_keys
            result['correct_answer_sanity_option_keys'] = (
                self.correct_answer_sanity_option_keys
            )

        if self.propositions is not None:
            result['propositions'] = self.propositions

        return result

    def is_student_correct(self) -> bool | None:
        """Evaluates whether the student's answer is correct.

        Determines correctness based on the question type:

        - **trueOrFalse**: ``student_answer == is_correct``
        - **checkboxes**: **all** options satisfy
          ``student_answer == is_correct``
        - **multiChoice**: ``selected_sanity_option_key ==
          correct_answer_sanity_option_key``

        Also updates ``self.correct`` with the computed value.

        Returns:
            ``True`` if the answer is correct, ``False`` if incorrect,
            or ``None`` if the student's answer is missing.

        Raises:
            ValueError: If the question type is not recognised.
        """
        if self.type == QuestionType.TRUE_OR_FALSE:
            if self.student_answer is None:
                self.correct = None
            else:
                self.correct = (self.student_answer == self.is_correct)

        elif self.type == QuestionType.CHECKBOXES:
            if self.options is None or any(
                opt.student_answer is None for opt in self.options
            ):
                self.correct = None
            else:
                self.correct = all(
                    opt.student_answer == opt.is_correct
                    for opt in self.options
                )

        elif self.type == QuestionType.MULTI_CHOICE:
            if self.selected_sanity_option_keys is None:
                self.correct = None
            else:
                self.correct = (
                    self.selected_sanity_option_keys
                    == self.correct_answer_sanity_option_keys
                )

        else:
            raise ValueError(f"Unknown question type: {self.type}")

        return self.correct


class Responses:
    """A collection of Response objects with indexing and statistics.

    Responses are indexed by both question id and student id for
    efficient lookups.

    Args:
        responses: List of Response objects.
    """

    def __init__(self, responses: list[Response]) -> None:
        self._responses = list(responses)
        self._by_text_question: dict[
            tuple[str, str], list[Response]
        ] = defaultdict(list)
        self._by_student: dict[str, list[Response]] = defaultdict(list)
        self._build_indexes()

    @classmethod
    def from_json(
        cls,
        filepath: str,
        valid_text_ids: set[str] | None = None,
    ) -> Responses:
        """Loads responses from a JSON file.

        The JSON file contains a list of student objects, each with a
        nested ``responses`` dict structured as
        ``text_id → question_id → [response, ...]``.

        Args:
            filepath: Path to the responses JSON file.
            valid_text_ids: If provided, only responses for these text
                ids are included.  If ``None``, all texts are included.

        Returns:
            A new Responses instance containing the loaded responses.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        responses: list[Response] = []
        for student in data:
            student_id = student.get('user_id')
            responses_dict = student.get('responses', {})

            for t_id, q_dict in responses_dict.items():
                if valid_text_ids is not None and t_id not in valid_text_ids:
                    continue
                for q_id, resp_list in q_dict.items():
                    if not isinstance(resp_list, list):
                        continue
                    for r in resp_list:
                        r['sanity_text_id'] = t_id
                        r['question_id'] = q_id
                        r['student_id'] = student_id
                        responses.append(Response.from_dict(r))

        return cls(responses)

    def to_json(self, filepath: str) -> None:
        """Saves responses to a JSON file.

        Writes responses in the nested format expected by
        ``from_json``: a list of student objects, each containing
        ``user_id`` and a ``responses`` dict structured as
        ``text_id → question_id → [response, ...]``.

        Only ``user_id`` and ``responses`` are included per student.

        Args:
            filepath: Path to write the JSON file to.
        """
        # Group responses by student → text → question
        students: dict[str, dict[str, dict[str, list]]] = {}
        for r in self._responses:
            student_texts = students.setdefault(r.student_id, {})
            text_questions = student_texts.setdefault(r.sanity_text_id, {})
            question_responses = text_questions.setdefault(
                r.question_id, []
            )
            question_responses.append(r.to_dict())

        data = [
            {'user_id': s_id, 'responses': texts}
            for s_id, texts in students.items()
        ]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _build_indexes(self) -> None:
        """Builds internal indexes by (text_id, question_id) and student id."""
        for r in self._responses:
            self._by_text_question[
                (r.sanity_text_id, r.question_id)
            ].append(r)
            self._by_student[r.student_id].append(r)

    def __len__(self) -> int:
        return len(self._responses)

    def __iter__(self):
        return iter(self._responses)

    def __getitem__(self, index):
        return self._responses[index]

    def get_by_text_question(
        self, text_id: str, question_id: str,
    ) -> list[Response]:
        """Returns all responses for a given text-question pair.

        Args:
            text_id: The text identifier to look up.
            question_id: The question identifier to look up.

        Returns:
            List of Response objects for that pair, or an empty list
            if the combination is not found.
        """
        return self._by_text_question.get((text_id, question_id), [])

    def get_by_student(self, student_id: str) -> list[Response]:
        """Returns all responses for a given student.

        Args:
            student_id: The student identifier to look up.

        Returns:
            List of Response objects for that student, or an empty
            list if the student id is not found.
        """
        return self._by_student.get(student_id, [])

    def remove_student_answers(self) -> None:
        """Clears student answer fields from all responses.

        For each response, removes the student's answer and resets
        ``correct`` to ``None``.  The specific fields cleared depend
        on the question type:

        - **trueOrFalse**: ``student_answer``
        - **checkboxes**: ``student_answer`` on each option
        - **multiChoice**: ``selected_sanity_option_key``

        This is intended to be called before simulation to ensure
        that the simulator does not have access to the real student
        answers.

        Raises:
            ValueError: If a response has an unrecognised question type.
        """
        for r in self._responses:
            r.correct = None
            if r.type == QuestionType.TRUE_OR_FALSE:
                r.student_answer = None
            elif r.type == QuestionType.CHECKBOXES:
                if r.options:
                    for opt in r.options:
                        opt.student_answer = None
            elif r.type == QuestionType.MULTI_CHOICE:
                r.selected_sanity_option_keys = None
            else:
                raise ValueError(f"Unknown question type: {r.type}")

    def correctness_by_text_question(
        self,
    ) -> dict[tuple[str, str], float]:
        """Returns the mean correctness rate per text-question pair.

        Returns:
            A dict mapping ``(text_id, question_id)`` to the fraction
            of correct responses for that pair.
        """
        result = {}
        for key, resps in self._by_text_question.items():
            total = len(resps)
            correct = sum(1 for r in resps if r.correct is True)
            result[key] = correct / total
        return result

    def correctness_by_student(self) -> dict[str, float]:
        """Returns the mean correctness rate per student.

        Returns:
            A dict mapping ``student_id`` to the fraction of correct
            responses for that student.
        """
        result = {}
        for s_id, resps in self._by_student.items():
            total = len(resps)
            correct = sum(1 for r in resps if r.correct is True)
            result[s_id] = correct / total
        return result

    def overall_accuracy(self) -> float:
        """Returns the overall fraction of correct responses.

        Returns:
            The fraction of responses where ``correct`` is ``True``,
            or ``0.0`` if the collection is empty.
        """
        if not self._responses:
            return 0.0
        correct_count = sum(
            1 for r in self._responses if r.correct is True
        )
        return correct_count / len(self._responses)
