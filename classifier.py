"""Single text-first classifier for four speech/audio categories."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import numpy as np

import config
from audio_utils import compute_rms, compute_snr_db, preprocess_for_noise, spectral_entropy, is_speech_with_vad

CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
LATIN_RE = re.compile(r"[a-z]", re.IGNORECASE)
LETTER_RE = re.compile(r"[а-яёa-z]", re.IGNORECASE)
WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)
GARBAGE_RE = re.compile(r"^[\W_\d]+$", re.UNICODE)

VOWELS_ONLY_RE = re.compile(r"^[аоыеиуэюяaoeuiy]+$", re.IGNORECASE)


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

    is_speech, speech_ratio, _ = is_speech_with_vad(samples, sample_rate)

    if speech_ratio < config.VAD_SPEECH_RATIO_LOW:
        return True, metrics

    if speech_ratio > config.VAD_SPEECH_RATIO_HIGH:
        return False, metrics

    effective_silence_threshold = config.RMS_SILENCE_THRESHOLD * 0.5

    if prepared is None:
        return False, metrics

    silent = rms <= effective_silence_threshold
    low_snr = snr <= config.SNR_NOISE_THRESHOLD_DB
    noise_like = entropy >= config.SPECTRAL_ENTROPY_NOISE_THRESHOLD
    hum_like = entropy <= config.SPECTRAL_ENTROPY_TONE_THRESHOLD and low_snr

    return bool(silent and low_snr), metrics


def classify_fragment(
    text: str,
    audio_samples: Optional[np.ndarray] = None,
    sample_rate: int = config.OUT_RATE,
    confidence: Optional[float] = None
) -> ClassificationResult:
    cleaned = normalize_text(text)
    audio_noise, metrics = is_audio_noise(audio_samples, sample_rate)
    bad_speech_audio = _looks_like_bad_speech_audio(metrics)

    _, speech_ratio, speech_duration = is_speech_with_vad(audio_samples, sample_rate)

    rms = metrics.get("rms", 0.0)

    # ----- ЗАЩИТЫ -----
    if audio_noise and rms <= config.RMS_SILENCE_THRESHOLD:
        if not cleaned or not CYRILLIC_RE.search(cleaned):
            return ClassificationResult(4, "silence with hallucinated text", **metrics)

    if audio_noise and len(cleaned) <= 12 and VOWELS_ONLY_RE.match(cleaned):
        return ClassificationResult(4, "vowel hallucination (noise)", **metrics)

    if LATIN_RE.search(cleaned) and len(cleaned) <= 2 and rms <= config.RMS_MUMBLE_THRESHOLD:
        return ClassificationResult(4, "very short latin noise", **metrics)

    if len(cleaned) == 1 and CYRILLIC_RE.match(cleaned):
        if audio_noise:
            return ClassificationResult(4, "single cyrillic letter on noise", **metrics)
        else:
            return ClassificationResult(3, "single cyrillic letter", **metrics)

    if not cleaned:
        return ClassificationResult(4, "empty text", **metrics)

    words = WORD_RE.findall(cleaned)
    letters = LETTER_RE.findall(cleaned)
    if not letters:
        return ClassificationResult(4, "no letters" if GARBAGE_RE.fullmatch(cleaned) else "symbols only", **metrics)

    garbage_ratio = 1.0 - (len("".join(letters)) / max(len(cleaned.replace(" ", "")), 1))
    if garbage_ratio >= config.GARBAGE_SYMBOL_RATIO:
        return ClassificationResult(4 if audio_noise else 3, "garbage symbols", **metrics)

    if LATIN_RE.search(cleaned):
        return _with_noise_overlay(2, "latin letters", audio_noise, metrics)

    if _is_laughter_only(words):
        return ClassificationResult(4, "laughter without distinguishable speech", **metrics)

    if CYRILLIC_RE.search(cleaned):
        is_short_text = len(cleaned) <= config.SHORT_TEXT_MAX_LEN
        word_count = len(words)

        # ----- ПРОВЕРКА ОЧЕНЬ КОРОТКОГО ТЕКСТА -----
        if word_count <= 3 and speech_duration > 0 and speech_duration < config.SHORT_SPEECH_DURATION_THRESHOLD_SEC:
            snr = metrics.get("snr_db", 0.0)
            if snr < 10.0:
                if audio_noise or speech_ratio < 0.3:
                    return ClassificationResult(4, "short low-duration noise", **metrics)
                else:
                    return ClassificationResult(3, "short low-duration speech", **metrics)
            else:
                if audio_noise:
                    return ClassificationResult(3, "short high-SNR but noisy", **metrics)

        # ----- КОРОТКИЕ СЛОВА (≤3 букв) -----
        if is_short_text:
            snr = metrics.get("snr_db", 0.0)
            has_hallucination = cleaned in config.HALLUCINATION_WORDS
            is_special = cleaned in config.SPECIAL_SHORT_WORDS

            # ----- СПЕЦИАЛЬНАЯ ОБРАБОТКА ДЛЯ "не", "да", "угу" и т.д. -----
            if is_special:
                # 1. Проверка уверенности
                if confidence is not None and confidence < config.SPECIAL_SHORT_WORD_CONFIDENCE_THRESHOLD:
                    if audio_noise:
                        return ClassificationResult(4, f"special word low confidence ({confidence:.2f}) on noise", **metrics)
                    else:
                        return ClassificationResult(3, f"special word low confidence ({confidence:.2f})", **metrics)

                # 2. Проверка длительности речи
                if speech_duration > 0 and speech_duration < config.SPECIAL_SHORT_WORD_MIN_DURATION:
                    if audio_noise:
                        return ClassificationResult(4, "special word too short on noise", **metrics)
                    else:
                        return ClassificationResult(3, "special word too short", **metrics)

                # 3. Проверка SNR и speech_ratio
                if snr < config.SPECIAL_SHORT_WORD_MIN_SNR or speech_ratio < config.SPECIAL_SHORT_WORD_MIN_SPEECH_RATIO:
                    if audio_noise:
                        return ClassificationResult(4, "special word low SNR/ratio on noise", **metrics)
                    else:
                        return ClassificationResult(3, "special word low SNR/ratio", **metrics)

                # Если прошло все проверки, считаем реальной речью
                return ClassificationResult(1, f"special word '{cleaned}' with good quality", **metrics)

            # ----- ОБЫЧНАЯ ЛОГИКА ДЛЯ ГАЛЛЮЦИНАЦИЙ -----
            if has_hallucination:
                if confidence is not None and confidence < config.SHORT_WORD_CONFIDENCE_THRESHOLD:
                    if audio_noise:
                        return ClassificationResult(4, f"hallucination low confidence ({confidence:.2f})", **metrics)
                    else:
                        return ClassificationResult(3, f"hallucination low confidence ({confidence:.2f})", **metrics)
                if snr > config.HALLUCINATION_SNR_THRESHOLD and speech_ratio > config.HALLUCINATION_SPEECH_RATIO:
                    return ClassificationResult(1, f"hallucination word with high SNR ({snr:.1f} dB)", **metrics)
                else:
                    if audio_noise or speech_ratio < 0.2:
                        return ClassificationResult(4, "hallucination word on noise", **metrics)
                    else:
                        return ClassificationResult(3, "hallucination word with poor quality", **metrics)

            # ----- ОБЫЧНАЯ ЛОГИКА ДЛЯ КОРОТКИХ СЛОВ (НЕ ГАЛЛЮЦИНАЦИИ) -----
            if confidence is not None and confidence < config.SHORT_WORD_CONFIDENCE_THRESHOLD:
                if audio_noise:
                    return ClassificationResult(4, f"short word low confidence ({confidence:.2f}) on noise", **metrics)
                else:
                    return ClassificationResult(3, f"short word low confidence ({confidence:.2f})", **metrics)

            if snr > config.SHORT_WORD_SNR_THRESHOLD and speech_ratio > config.SHORT_WORD_SPEECH_RATIO:
                return ClassificationResult(1, f"short word with good SNR ({snr:.1f} dB)", **metrics)
            elif snr > 3.0 and speech_ratio > 0.5:
                return ClassificationResult(1, f"short word with moderate SNR ({snr:.1f} dB)", **metrics)
            else:
                if audio_noise or speech_ratio < 0.2:
                    return ClassificationResult(4, "noise with short text", **metrics)
                else:
                    return ClassificationResult(3, "unintelligible short text with noise", **metrics)

        # ----- НЕ-КОРОТКИЕ СЛОВА -----
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