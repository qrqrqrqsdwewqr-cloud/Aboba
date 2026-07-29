"""Rule-based transcript postprocessing after GigaSTT without changing meaning."""
from __future__ import annotations

from dataclasses import dataclass
from text_utils import apply_replacements


@dataclass(frozen=True)
class PostprocessRules:
    user_replacements: dict[str, str]
    asr_replacements: dict[str, str]
    spoken_name_forms: dict[str, str]
    colloquial_replacements: dict[str, str]
    yo_replacements: dict[str, str]


class TranscriptPostprocessor:
    def __init__(self, rules: PostprocessRules):
        self.rules = rules

    def process(self, text: str) -> str:
        value = text or ""
        for mapping in (
            self.rules.asr_replacements,
            self.rules.spoken_name_forms,
            self.rules.colloquial_replacements,
            self.rules.user_replacements,
            self.rules.yo_replacements,
        ):
            value = apply_replacements(value, mapping)
        return value
