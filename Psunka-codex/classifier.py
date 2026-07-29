"""Single text-first classifier for four speech/audio categories."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import numpy as np

import config
from audio_utils import compute_rms, compute_snr_db, preprocess_for_noise, spectral_entropy

CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
LATIN_RE = re.compile(r"[a-z]", re.IGNORECASE)
LETTER_RE = re.compile(r"[а-яёa-z]", re.IGNORECASE)
WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)
GARBAGE_RE = re.compile(r"^[\W_\d]+$", re.UNICODE)


@dataclass(frozen=True)
class ClassificationResult:
    category: int
    reason: str
    rms: Optional[float] = None
    snr_db: Optional[float] = None
    spectral_entropy: Optional[float] = None
    categories: tuple[int, ...] | None = None

    @property
    def selected_categories(self) -> tuple[int, ...]:
        return self.categories or (self.category,)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def is_audio_noise(samples: Optional[np.ndarray], sample_rate: int) -> tuple[bool, dict[str, float]]:
    prepared = preprocess_for_noise(samples, sample_rate) if samples is not None else None
    rms = compute_rms(prepared)
    snr = compute_snr_db(prepared)
    entropy = spectral_entropy(prepared)
    metrics = {"rms": rms, "snr_db": snr, "spectral_entropy": entropy}
    if prepared is None:
        return False, metrics
    silent = rms <= config.RMS_SILENCE_THRESHOLD
    low_snr = snr <= config.SNR_NOISE_THRESHOLD_DB
    noise_like = entropy >= config.SPECTRAL_ENTROPY_NOISE_THRESHOLD
    hum_like = entropy <= config.SPECTRAL_ENTROPY_TONE_THRESHOLD and low_snr
    return bool(silent or (low_snr and noise_like) or hum_like), metrics


def classify_fragment(text: str, audio_samples: Optional[np.ndarray] = None, sample_rate: int = config.OUT_RATE, language: str | None = None) -> ClassificationResult:
    """Classify by text first; audio only promotes empty/garbage text to noise."""
    cleaned = normalize_text(text)
    audio_noise, metrics = is_audio_noise(audio_samples, sample_rate)
    bad_speech_audio = _looks_like_bad_speech_audio(metrics)

    if not cleaned:
        return ClassificationResult(4, "empty text", **metrics)

    words = WORD_RE.findall(cleaned)
    letters = LETTER_RE.findall(cleaned)
    if not letters:
        return ClassificationResult(4, "no letters" if GARBAGE_RE.fullmatch(cleaned) else "symbols only", **metrics)

    garbage_ratio = 1.0 - (len("".join(letters)) / max(len(cleaned.replace(" ", "")), 1))
    if garbage_ratio >= config.GARBAGE_SYMBOL_RATIO:
        return ClassificationResult(4 if audio_noise else 3, "garbage symbols", **metrics)

    if language in {"en", "kk", "ky", "uz"}:
        return _with_noise_overlay(2, f"detected language {language}", audio_noise, metrics)

    if LATIN_RE.search(cleaned):
        return _with_noise_overlay(2, "latin letters", audio_noise, metrics)

    if _is_laughter_only(words):
        return ClassificationResult(4, "laughter without distinguishable speech", **metrics)

    if CYRILLIC_RE.search(cleaned):
        if _looks_unintelligible(cleaned, words):
            return ClassificationResult(3, "fragmented/interjection text", **metrics)
        if config.DEGRADE_RUSSIAN_ON_BAD_AUDIO and bad_speech_audio:
            return ClassificationResult(3, "russian text with poor audio quality", **metrics)
        return _with_noise_overlay(1, "cyrillic russian-like words", audio_noise, metrics)

    return ClassificationResult(3, "fallback unintelligible", **metrics)


def _with_noise_overlay(category: int, reason: str, audio_noise: bool, metrics: dict[str, float]) -> ClassificationResult:
    if audio_noise and category in (1, 2, 3):
        return ClassificationResult(category, reason + " + background noise", **metrics, categories=(category, 4))
    return ClassificationResult(category, reason, **metrics, categories=(category,))


def _looks_unintelligible(text: str, words: list[str]) -> bool:
    if not words:
        return True
    if text.endswith("-") or "..." in text or "…" in text:
        return True
    if len(text) <= config.SHORT_FRAGMENT_MAX_CHARS and text not in {"да", "угу", "ага"}:
        return True
    marker_count = sum(1 for word in words if word in config.FRAGMENT_MARKERS)
    if marker_count == len(words) and not any(word in {"да", "угу", "ага"} for word in words):
        return True
    short_count = sum(1 for word in words if len(word) < config.MIN_RUSSIAN_WORD_LEN)
    return short_count > max(1, len(words) // 2)


def _looks_like_bad_speech_audio(metrics: dict[str, float]) -> bool:
    rms = metrics.get("rms", 0.0)
    snr = metrics.get("snr_db", 0.0)
    entropy = metrics.get("spectral_entropy", 1.0)
    if rms <= 0.0:
        return False
    very_quiet = rms <= config.RMS_MUMBLE_THRESHOLD
    low_snr = snr <= config.SNR_UNINTELLIGIBLE_THRESHOLD_DB
    broadband_noise = entropy >= config.SPECTRAL_ENTROPY_NOISE_THRESHOLD
    return bool((very_quiet and low_snr) or (low_snr and broadband_noise))


def _is_laughter_only(words: list[str]) -> bool:
    return bool(words) and all(word in config.LAUGHTER_MARKERS for word in words)
