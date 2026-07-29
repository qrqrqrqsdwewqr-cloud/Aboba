"""Audio buffering, timestamp matching, metrics, and filters."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

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
