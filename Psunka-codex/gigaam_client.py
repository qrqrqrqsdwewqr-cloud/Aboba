"""Local reusable GigaAM Multilingual client."""
from __future__ import annotations

from dataclasses import dataclass
import tempfile, wave
import os
import numpy as np

import config
from audio_utils import resample_audio, stereo_to_mono

@dataclass
class TranscriptionResult:
    text: str
    words: list
    duration_seconds: float
    model_name: str

class GigaAMClient:
    def __init__(self):
        self.model = None
        self.model_name = f"{config.GIGAAM_REPOSITORY}@{config.GIGAAM_REVISION}"

    def load(self) -> None:
        if self.model is not None: return
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(config.GIGAAM_REPOSITORY, revision=config.GIGAAM_REVISION, trust_remote_code=config.GIGAAM_TRUST_REMOTE_CODE)
        if hasattr(self.model, "to"):
            self.model = self.model.to(config.GIGAAM_DEVICE)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def transcribe(self, samples, sample_rate: int) -> TranscriptionResult:
        self.load()
        mono = stereo_to_mono(np.asarray(samples, dtype=np.float32))
        audio = resample_audio(mono, sample_rate, config.GIGAAM_INPUT_RATE).astype(np.float32, copy=False)
        duration = float(audio.size / config.GIGAAM_INPUT_RATE) if config.GIGAAM_INPUT_RATE else 0.0
        path = self._write_temp_wav(audio, config.GIGAAM_INPUT_RATE)
        try:
            try:
                raw = self.model.transcribe(path, word_timestamps=True)
            except TypeError:
                raw = self.model.transcribe(path)
            text, words = self._parse(raw)
            return TranscriptionResult(text=text, words=words, duration_seconds=duration, model_name=self.model_name)
        finally:
            try: os.remove(path)
            except OSError: pass

    def close(self) -> None:
        self.model = None

    @staticmethod
    def _parse(raw) -> tuple[str, list]:
        if isinstance(raw, str): return raw, []
        if isinstance(raw, dict): return str(raw.get("text") or raw.get("transcription") or ""), list(raw.get("words") or [])
        return str(raw or ""), []

    @staticmethod
    def _write_temp_wav(samples: np.ndarray, sample_rate: int) -> str:
        fd, path = tempfile.mkstemp(prefix="gigaam_", suffix=".wav"); os.close(fd)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(pcm.tobytes())
        return path
