"""Text normalization helpers for transcripts."""
from __future__ import annotations

import re


def apply_replacements(text: str, mapping: dict[str, str]) -> str:
    if not text or not mapping:
        return text
    keys = sorted(mapping, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(key) for key in keys) + r")\b"
    return re.sub(pattern, lambda match: _match_case(mapping.get(match.group(0).lower(), match.group(0)), match.group(0)), text, flags=re.IGNORECASE)


def apply_yo_replacements(text: str, mapping: dict[str, str]) -> str:
    return apply_replacements(text, mapping)


def normalize_transcript_text(text: str, replacements: dict[str, str], yo_replacements: dict[str, str]) -> str:
    return apply_yo_replacements(apply_replacements(text, replacements), yo_replacements)


def trim_repeated_prefix(text: str, previous_text: str, max_words: int) -> str:
    if not previous_text or not text:
        return text
    current = text.split()
    previous = previous_text.split()
    limit = min(max_words, len(current), len(previous))
    for size in range(limit, 0, -1):
        if previous[-size:] == current[:size]:
            return " ".join(current[size:]).strip()
    return text


def _match_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement
