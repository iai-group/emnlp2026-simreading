"""Question type dataclasses and question-loading utilities.

Provides typed representations for the different question formats used in
the SimReading reading comprehension dataset, as well as a ``Questions``
loader class that indexes the raw ``questions.json`` file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from simreading.text import Text

QUESTIONS_JSON_FILE = 'data/questions.json'

class QuestionTypeEnum(str, Enum):
    """Enumeration of supported question types."""

    TRUE_OR_FALSE = 'trueOrFalse'
    CHECKBOXES = 'checkboxes'
    MULTI_CHOICE = 'multiChoice'
    FEEDBACK = 'feedback'
    FREE_TEXT = 'freeText'


# Backward-compatible alias used by other modules
QuestionType = QuestionTypeEnum


@dataclass
class AnswerOption:
    """A single answer option within a multiple-choice question.

    Attributes:
        sanity_answer_key: Original Sanity CMS key for this answer.
        option: The answer text.
        is_correct: Whether this option is the correct answer.
    """

    sanity_answer_key: str
    option: str
    is_correct: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> AnswerOption:
        """Creates an ``AnswerOption`` from a raw JSON dictionary."""
        return cls(
            sanity_answer_key=data['sanity_answer_key'],
            option=data.get('option', ''),
            is_correct=data.get('is_correct', False),
        )

    def to_dict(self) -> dict:
        """Serializes this option to a JSON-compatible dictionary."""
        return {
            'sanity_answer_key': self.sanity_answer_key,
            'option': self.option,
            'is_correct': self.is_correct,
        }



@dataclass
class Question:
    """Base class for all question types.

    Attributes:
        id: Internal identifier (unique across the dataset).
        type: The question type string (e.g. ``'trueOrFalse'``).
        sanity_text_id: Sanity CMS key for the parent text.
        sanity_question_key: Original Sanity CMS key for this question.
        question: The question text.
    """

    id: str
    type: str
    sanity_text_id: str
    sanity_question_key: str
    question: str
    category: str | None = None

    def to_dict(self) -> dict:
        """Serializes this question to a JSON-compatible dictionary."""
        return {
            'question_id': self.id,
            'sanity_question_key': self.sanity_question_key,
            'type': self.type,
            'question': self.question,
            'category': self.category,
        }

    def render_question(self) -> str:
        """Returns the question text formatted for inclusion in a prompt."""
        return self.question

    def render_options(self) -> str:
        """Returns the answer options formatted for inclusion in a prompt.

        Must be overridden by subclasses that support rendering.
        """
        raise NotImplementedError(
            f"render_options() is not implemented for base Question class. "
            f"Use a specific subclass instead."
        )

    def render(self) -> str:
        """Returns both the question and its options as a single string."""
        parts = [self.render_question()]
        options_text = self.render_options()
        if options_text:
            parts.append(options_text)
        return '\n'.join(parts)



@dataclass
class TrueOrFalse(Question):
    """A true-or-false question representing a single statement.

    In the raw data a true-or-false question contains multiple options
    (statements).  Each statement is converted into its own
    ``TrueOrFalse`` instance by the conversion script, so this class
    always represents exactly *one* statement the respondent must judge
    as true or false.  The statement text is stored in ``question``.

    Attributes:
        type: Always ``'trueOrFalse'``.
        sanity_answer_key: Sanity CMS key for this answer option.
        is_correct: Whether the correct answer is "true".
    """

    type: str = field(default='trueOrFalse', init=False)
    sanity_answer_key: str = ''
    is_correct: bool = False

    def to_dict(self) -> dict:
        """Serializes this question to a JSON-compatible dictionary."""
        d = super().to_dict()
        d['sanity_answer_key'] = self.sanity_answer_key
        d['is_correct'] = self.is_correct
        return d

    def render_question(self) -> str:
        """Returns the question text formatted for inclusion in a prompt."""
        return f"Er følgende påstand rett eller galt?\n{self.question}"

    def render_options(self) -> str:
        """Renders the True/False answer choices."""
        return '  ( ) Rett\n  ( ) Galt'



@dataclass
class Checkboxes(Question):
    """A checkbox (select-all-that-apply) question.

    Contains multiple options that the respondent may independently
    select or leave unselected.  Each option is identified by its
    ``sanity_answer_key``.

    Attributes:
        type: Always ``'checkboxes'``.
        options: The list of answer options.
    """

    type: str = field(default='checkboxes', init=False)
    options: Sequence[AnswerOption] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serializes this question to a JSON-compatible dictionary."""
        d = super().to_dict()
        d['options'] = [opt.to_dict() for opt in self.options]
        return d

    def get_option_keys(self) -> list[str]:
        """Returns the ``sanity_answer_key`` for every option."""
        return [opt.sanity_answer_key for opt in self.options]

    def render_options(self) -> str:
        """Renders each option with a checkbox marker."""
        lines: list[str] = []
        for opt in self.options:
            lines.append(f'  [ ] {opt.option}')
        return '\n'.join(lines)



@dataclass
class MultiChoice(Question):
    """A multiple-choice question (pick one).

    There is a single question with several answer options, each
    identified by its ``sanity_answer_key``.

    Attributes:
        type: Always ``'multiChoice'``.
        options: The list of answer options.
    """

    type: str = field(default='multiChoice', init=False)
    options: Sequence[AnswerOption] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serializes this question to a JSON-compatible dictionary."""
        d = super().to_dict()
        d['options'] = [opt.to_dict() for opt in self.options]
        return d

    def get_option_keys(self) -> list[str]:
        """Returns the ``sanity_answer_key`` for every option."""
        return [opt.sanity_answer_key for opt in self.options]

    def render_options(self) -> str:
        """Renders options as a lettered list (A, B, C, …)."""
        lines: list[str] = []
        for i, opt in enumerate(self.options):
            letter = chr(65 + i)
            lines.append(f'  ({letter}) {opt.option}')
        return '\n'.join(lines)



@dataclass
class Feedback(Question):
    """A feedback question.

    This question type is not supported for prompt rendering and will
    raise :class:`NotImplementedError` if :meth:`render_options` or
    :meth:`render` is called.

    Attributes:
        type: Always ``'feedback'``.
    """

    type: str = field(default='feedback', init=False)

    def render_options(self) -> str:
        raise NotImplementedError(
            "Rendering is not supported for Feedback questions."
        )

    def render(self) -> str:
        raise NotImplementedError(
            "Rendering is not supported for Feedback questions."
        )



@dataclass
class FreeText(Question):
    """A free-text question.

    This question type is not supported for prompt rendering and will
    raise :class:`NotImplementedError` if :meth:`render_options` or
    :meth:`render` is called.

    Attributes:
        type: Always ``'freeText'``.
    """

    type: str = field(default='freeText', init=False)

    def render_options(self) -> str:
        raise NotImplementedError(
            "Rendering is not supported for FreeText questions."
        )

    def render(self) -> str:
        raise NotImplementedError(
            "Rendering is not supported for FreeText questions."
        )



_TYPE_MAP: dict[str, type[Question]] = {
    'trueOrFalse': TrueOrFalse,
    'checkboxes': Checkboxes,
    'multiChoice': MultiChoice,
    'feedback': Feedback,
    'freeText': FreeText,
}



def question_from_dict(
    data: dict,
    sanity_text_id: str,
) -> Question:
    """Creates a typed :class:`Question` instance from a raw JSON dict.

    Args:
        data: A single question dictionary from ``questions.json``.
        sanity_text_id: The ``sanity_text_id`` of the parent text.

    Returns:
        A single :class:`Question` instance.

    Raises:
        ValueError: If the question type is not recognized.
    """
    q_type = data.get('type', '')
    cls = _TYPE_MAP.get(q_type)
    if cls is None:
        raise ValueError(f"Unknown question type: {q_type!r}")

    sanity_question_key = data['sanity_question_key']
    question_text = data.get('question', '')
    category = data.get('category')

    # TrueOrFalse: each dict is already a single statement
    if cls is TrueOrFalse:
        return cls(
            id=data.get('question_id', f"{sanity_question_key}-{data['sanity_answer_key']}"),
            sanity_text_id=sanity_text_id,
            sanity_question_key=sanity_question_key,
            question=question_text,
            sanity_answer_key=data['sanity_answer_key'],
            is_correct=data.get('is_correct', False),
            category=category,
        )

    # MultiChoice / Checkboxes: single question with options
    if cls in (MultiChoice, Checkboxes):
        options = [
            AnswerOption.from_dict(opt)
            for opt in data.get('options', [])
        ]
        return cls(
            id=data.get('question_id', sanity_question_key),
            sanity_text_id=sanity_text_id,
            sanity_question_key=sanity_question_key,
            question=question_text,
            options=options,
            category=category,
        )

    # Feedback / FreeText: no options
    return cls(
        id=data.get('question_id', sanity_question_key),
        sanity_text_id=sanity_text_id,
        sanity_question_key=sanity_question_key,
        question=question_text,
        category=category,
    )



class Questions:
    """Provides indexed access to questions loaded from a JSON file or
    pre-built :class:`Text` objects.

    Attributes:
        text_entries: List of all :class:`Text` instances.
    """

    def __init__(
        self,
        filepath: str | None = QUESTIONS_JSON_FILE,
        ignore_free_text_questions: bool = True,
        text_entries: list[Text] | None = None,
    ):
        """Initialises by loading and indexing questions.

        Supply either *filepath* (to load from JSON) **or**
        *text_entries* (pre-built objects).  If both are given,
        *text_entries* takes precedence.

        Args:
            filepath: Path to the questions JSON file.
            ignore_free_text_questions: If True, questions with type
                ``"freeText"`` are excluded from the indexes.
            text_entries: Pre-built :class:`Text` list.  When
                provided the file is not read.
        """
        self._ignore_free_text = ignore_free_text_questions

        if text_entries is not None:
            self.text_entries = text_entries
        else:
            if filepath and not os.path.exists(filepath):
                default_path = os.path.join(
                    os.path.dirname(__file__), '..', 'data', 'questions.json'
                )
                filepath = os.path.abspath(default_path)

            with open(filepath, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            self.text_entries = self._parse_json(raw_data)

        self._build_indexes()

    def _parse_json(self, raw_data: list[dict]) -> list[Text]:
        """Converts raw JSON data into :class:`Text` objects.

        Text entries with ``ignore_text: true`` are skipped entirely;
        they are not added to the indexes and their questions are not
        loaded.
        """
        from simreading.text import Text
        entries: list[Text] = []
        for text in raw_data:
            if text.get('ignore_text', False):
                continue
            text_id = text['sanity_text_id']
            questions: list[Question] = []
            for q_dict in text.get('questions', []):
                if (self._ignore_free_text
                        and q_dict.get('type') == 'freeText'):
                    continue
                try:
                    questions.append(question_from_dict(q_dict, text_id))
                except ValueError:
                    pass
            entries.append(Text(
                sanity_text_id=text_id,
                serial_number=text.get('serialNumber', ''),
                title=text.get('title', ''),
                text=text.get('text', ''),
                questions=questions,
            ))
        return entries

    def _build_indexes(self):
        """Builds lookup dictionaries from :attr:`text_entries`."""
        self.texts_by_id: dict[str, Text] = {}
        self.questions_by_id: dict[str, Question] = {}

        for entry in self.text_entries:
            self.texts_by_id[entry.sanity_text_id] = entry
            for q in entry.questions:
                self.questions_by_id[q.id] = q

    def get_text(self, text_id: str) -> Text | None:
        """Returns the Text object for the given text id."""
        return self.texts_by_id.get(text_id)

    def get_question(self, question_id: str) -> Question | None:
        """Returns the Question object for the given question id."""
        return self.questions_by_id.get(question_id)

    def get_all_texts(self) -> list[Text]:
        """Returns all text entries."""
        return list(self.texts_by_id.values())

    def get_all_questions(self) -> list[Question]:
        """Returns all question entries."""
        return list(self.questions_by_id.values())

    def get_all_text_ids(self) -> set[str]:
        """Returns the set of all text ids."""
        return set(self.texts_by_id.keys())

    def to_json(self) -> list[dict]:
        """Serializes all text entries to a JSON-compatible list."""
        return [entry.to_dict() for entry in self.text_entries]

    def save_to_json(self, filepath: str = QUESTIONS_JSON_FILE) -> None:
        """Writes all text entries to a JSON file.

        Args:
            filepath: Destination file path.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, indent=4, ensure_ascii=False)
