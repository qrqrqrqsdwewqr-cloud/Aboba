"""Audio buffering, timestamp matching, metrics, and filters."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, resample_poly

import config


@dataclass(frozen=True)
class AudioFragment:
    timestamp: float
    samples: np.ndarray
    sample_rate: int


class AudioFragmentStore:
    """Thread-safe timestamped audio fragment store."""

    def __init__(self, max_fragments: int = config.AUDIO_STORE_MAX_FRAGMENTS):
        self._max_fragments = max_fragments
        self._items: list[AudioFragment] = []
        self._lock = threading.Lock()

    def add(self, samples: np.ndarray, sample_rate: int, timestamp: Optional[float] = None) -> AudioFragment:
        fragment = AudioFragment(timestamp or time.time(), samples.astype(np.float32).copy(), sample_rate)
        with self._lock:
            self._items.append(fragment)
            if len(self._items) > self._max_fragments:
                self._items = self._items[-self._max_fragments :]
        return fragment

    def nearest(self, timestamp: float, window_seconds: float = config.AUDIO_MATCH_WINDOW_SECONDS) -> Optional[AudioFragment]:
        with self._lock:
            if not self._items:
                return None
            best = min(self._items, key=lambda item: abs(item.timestamp - timestamp))
        return best if abs(best.timestamp - timestamp) <= window_seconds else None


def stereo_to_mono(data: np.ndarray) -> np.ndarray:
    if data is None:
        return np.array([], dtype=np.float32)
    if data.ndim == 1:
        return data.astype(np.float32).reshape(-1)
    if data.ndim == 2 and data.shape[1] == 1:
        return data.astype(np.float32).reshape(-1)
    return data.astype(np.float32).mean(axis=1)


def resample_audio(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if samples is None or samples.size == 0:
        return np.array([], dtype=np.float32)
    if src_rate == dst_rate:
        return samples.astype(np.float32)
    gcd = np.gcd(src_rate, dst_rate)
    return resample_poly(samples, dst_rate // gcd, src_rate // gcd).astype(np.float32)


def float32_to_int16_bytes(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def compute_rms(samples: Optional[np.ndarray]) -> float:
    if samples is None or samples.size == 0:
        return 0.0
    x = samples.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(x))))


def compute_snr_db(samples: Optional[np.ndarray], percentile: int = config.NOISE_FLOOR_PERCENTILE) -> float:
    """Estimate SNR from frame RMS distribution without VAD."""
    if samples is None or samples.size == 0:
        return 0.0
    x = samples.astype(np.float32)
    frame = max(1, min(len(x), 1024))
    usable = len(x) - (len(x) % frame)
    if usable <= 0:
        return 0.0
    frames = x[:usable].reshape(-1, frame)
    rms_values = np.sqrt(np.mean(np.square(frames), axis=1))
    signal = float(np.percentile(rms_values, 95))
    noise = max(float(np.percentile(rms_values, percentile)), 1e-9)
    return float(20.0 * np.log10(max(signal, 1e-9) / noise))


def spectral_entropy(samples: Optional[np.ndarray]) -> float:
    if samples is None or samples.size == 0:
        return 1.0
    x = samples.astype(np.float32)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    total = float(np.sum(spectrum))
    if total <= 1e-12:
        return 1.0
    p = spectrum / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(len(spectrum)))


# ===== НОВАЯ ФУНКЦИЯ СПЕКТРАЛЬНОГО ЦЕНТРОИДА =====
def spectral_centroid(samples: Optional[np.ndarray], sample_rate: int) -> float:
    """
    Вычисляет спектральный центроид (средневзвешенную частоту) в Гц.
    Для речи типичные значения: 1000–5000 Гц.
    """
    if samples is None or len(samples) == 0:
        return 0.0
    # Ограничиваем длину для ускорения вычислений
    if len(samples) > 32768:
        samples = samples[:32768]
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
    total = np.sum(spectrum)
    if total < 1e-12:
        return 0.0
    centroid = np.sum(freqs * spectrum) / total
    return float(centroid)


def bandpass_filter(samples: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    if samples is None or samples.size == 0:
        return np.array([], dtype=np.float32)
    nyquist = sample_rate / 2.0
    low = max(low_hz / nyquist, 1e-5)
    high = min(high_hz / nyquist, 0.99999)
    if low >= high:
        return samples.astype(np.float32)
    b, a = butter(4, [low, high], btype="band")
    return lfilter(b, a, samples).astype(np.float32)


def notch_filter(samples: np.ndarray, sample_rate: int, freq_hz: float, q: float) -> np.ndarray:
    if samples is None or samples.size == 0 or freq_hz <= 0:
        return np.array([], dtype=np.float32)
    b, a = iirnotch(freq_hz, q, sample_rate)
    return lfilter(b, a, samples).astype(np.float32)


def preprocess_for_noise(samples: Optional[np.ndarray], sample_rate: int) -> Optional[np.ndarray]:
    if samples is None:
        return None
    out = samples.astype(np.float32)
    if config.ENABLE_BANDPASS_FILTER:
        out = bandpass_filter(out, sample_rate, config.BANDPASS_LOW_HZ, config.BANDPASS_HIGH_HZ)
    if config.ENABLE_NOTCH_FILTER:
        out = notch_filter(out, sample_rate, config.NOTCH_FREQ_HZ, config.NOTCH_Q)
    return out


# ---------- ГЛОБАЛЬНЫЙ КЭШ МОДЕЛИ SILERO VAD ----------
_SILERO_MODEL = None


def _get_silero_model():
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        try:
            from silero_vad import load_silero_vad
            _SILERO_MODEL = load_silero_vad()
            print("Silero VAD модель успешно загружена")
        except Exception as e:
            print(f"Не удалось загрузить Silero VAD: {e}. Будет использован WebRTC VAD как fallback.")
            _SILERO_MODEL = False  # помечаем как недоступный
    return _SILERO_MODEL


def _webRTC_vad_fallback(samples: np.ndarray, sample_rate: int) -> Tuple[bool, float, float]:
    """Fallback на WebRTC VAD, если Silero недоступен. Возвращает (is_speech, speech_ratio, 0.0)."""
    try:
        import webrtcvad
        int16_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        if sample_rate != config.VAD_SAMPLE_RATE:
            from scipy.signal import resample_poly
            gcd = np.gcd(sample_rate, config.VAD_SAMPLE_RATE)
            int16_samples = resample_poly(int16_samples, config.VAD_SAMPLE_RATE // gcd, sample_rate // gcd).astype(np.int16)
        vad = webrtcvad.Vad(config.WEBRTC_AGRESSIVENESS)
        frame_duration_ms = config.WEBRTC_FRAME_DURATION_MS
        frame_bytes = int(config.VAD_SAMPLE_RATE * frame_duration_ms / 1000) * 2
        pcm_bytes = int16_samples.tobytes()
        speech_frames = 0
        total_frames = 0
        for i in range(0, len(pcm_bytes), frame_bytes):
            frame = pcm_bytes[i:i+frame_bytes]
            if len(frame) < frame_bytes:
                break
            try:
                if vad.is_speech(frame, config.VAD_SAMPLE_RATE):
                    speech_frames += 1
            except Exception:
                pass
            total_frames += 1
        ratio = speech_frames / total_frames if total_frames > 0 else 0.0
        is_speech = ratio > config.VAD_SPEECH_RATIO_THRESHOLD
        return is_speech, ratio, 0.0
    except Exception as e:
        print(f"WebRTC VAD fallback также не удался: {e}")
        return True, 1.0, 0.0  # в случае полной ошибки считаем речью


# ---------- ОСНОВНАЯ ФУНКЦИЯ VAD (Silero + fallback) С ВОЗВРАТОМ ДЛИТЕЛЬНОСТИ ----------
def is_speech_with_vad(samples: Optional[np.ndarray], sample_rate: int) -> Tuple[bool, float, float]:
    """
    Определяет наличие речи в аудио с помощью Silero VAD (основной) и WebRTC (fallback).
    Возвращает (is_speech, speech_ratio, total_speech_duration_sec)
    """
    if samples is None or samples.size == 0:
        return False, 0.0, 0.0

    model = _get_silero_model()
    if model is False:
        is_speech, ratio, _ = _webRTC_vad_fallback(samples, sample_rate)
        return is_speech, ratio, 0.0

    if model is None:
        model = _get_silero_model()
        if model is False or model is None:
            is_speech, ratio, _ = _webRTC_vad_fallback(samples, sample_rate)
            return is_speech, ratio, 0.0

    try:
        from silero_vad import get_speech_timestamps

        audio_float = samples.astype(np.float32)
        if sample_rate != config.VAD_SAMPLE_RATE:
            from scipy.signal import resample_poly
            gcd = np.gcd(sample_rate, config.VAD_SAMPLE_RATE)
            audio_float = resample_poly(audio_float, config.VAD_SAMPLE_RATE // gcd, sample_rate // gcd).astype(np.float32)

        speech_timestamps = get_speech_timestamps(
            audio_float,
            model,
            threshold=config.SILERO_VAD_THRESHOLD,
            min_speech_duration_ms=config.SILERO_MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=config.SILERO_MIN_SILENCE_DURATION_MS,
        )

        total_speech_ms = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
        total_duration_ms = len(audio_float) / config.VAD_SAMPLE_RATE * 1000

        speech_ratio = total_speech_ms / total_duration_ms if total_duration_ms > 0 else 0.0
        is_speech = speech_ratio > config.SILERO_SPEECH_RATIO_THRESHOLD
        speech_duration_sec = total_speech_ms / 1000.0

        return is_speech, speech_ratio, speech_duration_sec

    except Exception as e:
        print(f"Ошибка при использовании Silero VAD: {e}. Переключаемся на WebRTC fallback.")
        is_speech, ratio, _ = _webRTC_vad_fallback(samples, sample_rate)
        return is_speech, ratio, 0.0