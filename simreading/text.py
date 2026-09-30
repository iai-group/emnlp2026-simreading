"""Reading comprehension text dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field

from simreading.questions import Question


@dataclass
class Text:
    """A reading comprehension text with associated questions.

    Attributes:
        sanity_text_id: Sanity CMS key for this text.
        serial_number: Serial identifier (e.g. ``'HT_105_Ørner'``).
        title: Human-readable title.
        text: The full text content (Markdown).
        questions: The questions associated with this text.
        ignore_text: If True, this text and its questions/responses
            should be excluded from statistics, simulation, and
            evaluation. Defaults to False.
    """

    sanity_text_id: str
    serial_number: str
    title: str
    text: str
    questions: list[Question] = field(default_factory=list)
    ignore_text: bool = False

    def to_dict(self) -> dict:
        """Serializes this text to a JSON-compatible dictionary."""
        d = {
            'sanity_text_id': self.sanity_text_id,
            'serialNumber': self.serial_number,
            'title': self.title,
            'text': self.text,
            'questions': [q.to_dict() for q in self.questions],
        }
        if self.ignore_text:
            d['ignore_text'] = True
        return d
