"""Clipboard helpers kept separate from site automation."""
from __future__ import annotations


def copy_transcript(text: str) -> bool:
    if not (text or "").strip():
        return False
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False
