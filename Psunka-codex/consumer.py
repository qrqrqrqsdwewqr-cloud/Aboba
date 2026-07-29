"""Legacy JSON consumer helpers retained for non-main tests; no site automation."""
from __future__ import annotations

import json, queue, time, asyncio
from typing import Any

import config
from audio_utils import AudioFragmentStore
from classifier import classify_fragment
from clipboard_utils import copy_transcript
from postprocessing import PostprocessRules, TranscriptPostprocessor
from text_utils import trim_repeated_prefix

class TranscriptAggregator:
    def __init__(self, gui_queue: queue.Queue, log=print):
        self.gui_queue = gui_queue; self.log = log; self._pending: dict[float, dict[str, Any]] = {}; self._tasks: dict[float, asyncio.Task] = {}; self._last_emitted_text = ""
    def submit(self, fragment_key: float, text: str, audio) -> None:
        bucket = self._pending.setdefault(fragment_key, {"texts": [], "audio": audio}); bucket["audio"] = audio or bucket["audio"]; self._add_unique_text(bucket["texts"], text)
        task = self._tasks.pop(fragment_key, None)
        if task: task.cancel()
        self._tasks[fragment_key] = asyncio.create_task(self._flush_later(fragment_key))
    async def flush_all(self) -> None:
        for task in self._tasks.values(): task.cancel()
        for key in list(self._pending): self._flush(key)
    async def _flush_later(self, fragment_key: float) -> None:
        try: await asyncio.sleep(config.TRANSCRIPT_FLUSH_DELAY_SECONDS); self._flush(fragment_key)
        except asyncio.CancelledError: return
    def _flush(self, fragment_key: float) -> None:
        bucket = self._pending.pop(fragment_key, None); self._tasks.pop(fragment_key, None)
        if not bucket: return
        audio = bucket["audio"]; text_proc = self._trim_repeated_prefix(" ".join(bucket["texts"]).strip())
        if not text_proc: return
        result = classify_fragment(text_proc, audio.samples if audio else None, audio.sample_rate if audio else config.OUT_RATE)
        if result.category == 1: copy_transcript(text_proc)
        display_text = text_proc if result.category == 1 else _non_russian_placeholder(result.category)
        try: self.gui_queue.put_nowait((display_text or "<<empty>>", result)); self._last_emitted_text = text_proc
        except queue.Full: self.log("GUI queue full")
    @staticmethod
    def _add_unique_text(items: list[str], text: str) -> None:
        normalized = text.strip()
        if normalized and not any(normalized == item or normalized in item for item in items):
            items[:] = [item for item in items if item not in normalized]; items.append(normalized)
    def _trim_repeated_prefix(self, text: str) -> str:
        return trim_repeated_prefix(text, self._last_emitted_text, config.MAX_PREFIX_DEDUP_WORDS)

def extract_text(payload: dict[str, Any]) -> str | None:
    for key in ("text", "transcript", "result", "final"):
        value = payload.get(key)
        if isinstance(value, str): return value
    return None

def extract_timestamp(payload: dict[str, Any]) -> float:
    for key in ("timestamp", "ts", "time", "created_at"):
        value = payload.get(key)
        if isinstance(value, (int, float)): return float(value)
    return time.time()

async def consumer(ws, gui_queue: queue.Queue, audio_store: AudioFragmentStore, log=print) -> None:
    aggregator = TranscriptAggregator(gui_queue, log)
    try:
        async for msg in ws:
            if isinstance(msg, bytes): continue
            try: parsed = json.loads(msg)
            except Exception: log("Failed to parse JSON from server"); continue
            text = extract_text(parsed)
            if text is None: log("Server JSON has no text field: " + str(parsed)); continue
            audio = audio_store.nearest(extract_timestamp(parsed)); key = audio.timestamp if audio else _timestamp_bucket(time.time())
            post = TranscriptPostprocessor(PostprocessRules(config.REPLACEMENTS, config.ASR_REPLACEMENTS, config.SPOKEN_NAME_FORMS, config.COLLOQUIAL_REPLACEMENTS, config.YO_WORD_REPLACEMENTS))
            aggregator.submit(key, post.process(text), audio)
    finally: await aggregator.flush_all()

def _timestamp_bucket(timestamp: float) -> float:
    return timestamp - (timestamp % config.CHUNK_SECONDS)

def _non_russian_placeholder(category: int) -> str:
    return {2: "<<foreign speech: transcript hidden>>", 3: "<<unintelligible speech>>", 4: "<<noise>>"}.get(category, "<<unknown>>")
