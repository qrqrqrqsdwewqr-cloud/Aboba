"""Minimal text-only language detection for GigaAM transcripts."""
from __future__ import annotations

from dataclasses import dataclass
import re

@dataclass(frozen=True)
class LanguageResult:
    language: str
    is_russian: bool

RU = set("ёъыэ")
KK = set("әғқңөұүһі")
KY = set("ңүө")
UZ = set("ўқғҳ")
CYR = re.compile(r"[а-яёәғқңөұүһіўҳ]", re.I)
LAT = re.compile(r"[a-z]", re.I)

def detect_language(text: str) -> LanguageResult:
    value = (text or "").strip().lower()
    letters = re.findall(r"[a-zа-яёәғқңөұүһіўҳ]", value, re.I)
    if len(letters) < 4: return LanguageResult("unknown", False)
    joined = "".join(letters)
    has_cyr = bool(CYR.search(joined)); has_lat = bool(LAT.search(joined))
    if has_cyr and has_lat: return LanguageResult("unknown", False)
    if has_lat: return LanguageResult("en", False)
    chars = set(joined)
    if chars & UZ: return LanguageResult("uz", False)
    if chars & KK: return LanguageResult("kk", False)
    if chars & KY and not chars & RU: return LanguageResult("ky", False)
    if CYR.search(joined): return LanguageResult("ru", True)
    return LanguageResult("unknown", False)
