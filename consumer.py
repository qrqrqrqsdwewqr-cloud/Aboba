"""WebSocket consumer logic: timestamp matching, classification, GUI output, clipboard copy."""
from __future__ import annotations

import json
import queue
import time
import asyncio
from typing import Any

import config
from automation import press_category
from audio_utils import AudioFragmentStore
from classifier import classify_fragment, ClassificationResult
from postprocessing import PostprocessRules, TranscriptPostprocessor
from text_utils import trim_repeated_prefix

TRANSCRIPT_TIMEOUT = 3.0


class TranscriptAggregator:
    def __init__(self, gui_queue: queue.Queue, log=print):
        self.gui_queue = gui_queue
        self.log = log
        self._pending: dict[float, dict[str, Any]] = {}
        self._tasks: dict[float, asyncio.Task] = {}
        self._last_emitted_text = ""
        self._timeout_tasks: dict[float, asyncio.Task] = {}

    def start_waiting(self, fragment_key: float, audio) -> None:
        if fragment_key in self._pending:
            return
        self._pending[fragment_key] = {
            "texts": [],
            "confidences": [],
            "audio": audio,
            "processed": False,
        }
        self._start_timeout(fragment_key)
        self.log(f"Начато ожидание для фрагмента {fragment_key}")

    def submit(self, fragment_key: float, text: str, audio, confidence: float | None = None) -> None:
        if fragment_key not in self._pending:
            self._pending[fragment_key] = {
                "texts": [],
                "confidences": [],
                "audio": audio,
                "processed": False,
            }

        bucket = self._pending[fragment_key]
        bucket["audio"] = audio or bucket["audio"]
        self._add_unique_text(bucket["texts"], text)
        if confidence is not None:
            bucket["confidences"].append(confidence)

        timeout_task = self._timeout_tasks.pop(fragment_key, None)
        if timeout_task:
            timeout_task.cancel()
            self.log(f"Таймаут отменён для фрагмента {fragment_key} (получен текст)")

        task = self._tasks.pop(fragment_key, None)
        if task:
            task.cancel()
        self._tasks[fragment_key] = asyncio.create_task(self._flush_later(fragment_key))

    def _start_timeout(self, fragment_key: float) -> None:
        old = self._timeout_tasks.pop(fragment_key, None)
        if old:
            old.cancel()
        task = asyncio.create_task(self._timeout_handler(fragment_key))
        self._timeout_tasks[fragment_key] = task

    async def _timeout_handler(self, fragment_key: float) -> None:
        try:
            await asyncio.sleep(TRANSCRIPT_TIMEOUT)
            if fragment_key in self._pending:
                bucket = self._pending[fragment_key]
                if bucket.get("processed", False):
                    self.log(f"Таймаут: фрагмент {fragment_key} уже обработан, пропускаем")
                    return
                if not bucket.get("texts"):
                    self.log(f"Таймаут: нет ответа от сервера для фрагмента {fragment_key}, отправляем шум")
                    task = self._tasks.pop(fragment_key, None)
                    if task:
                        task.cancel()
                    self._pending.pop(fragment_key, None)
                    audio = bucket.get("audio")
                    result = ClassificationResult(4, "timeout - no response", 0.0, 0.0, 1.0)
                    display_text = "<<шум>>"
                    try:
                        self.gui_queue.put_nowait((display_text, result))
                        self._last_emitted_text = ""
                        self.log("Результат (таймаут) отправлен в GUI")
                    except queue.Full:
                        self.log("GUI queue full")
        except asyncio.CancelledError:
            self.log(f"Таймаут для фрагмента {fragment_key} был отменён")
            return

    async def flush_all(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        for task in self._timeout_tasks.values():
            task.cancel()
        for key in list(self._pending):
            self._flush(key)

    async def _flush_later(self, fragment_key: float) -> None:
        try:
            await asyncio.sleep(config.TRANSCRIPT_FLUSH_DELAY_SECONDS)
            self._flush(fragment_key)
        except asyncio.CancelledError:
            return

    def _flush(self, fragment_key: float) -> None:
        bucket = self._pending.pop(fragment_key, None)
        if not bucket:
            return

        timeout_task = self._timeout_tasks.pop(fragment_key, None)
        if timeout_task:
            timeout_task.cancel()

        task = self._tasks.pop(fragment_key, None)
        if task:
            task.cancel()

        bucket["processed"] = True

        audio = bucket["audio"]
        raw_text = " ".join(bucket["texts"]).strip()
        text_proc = self._trim_repeated_prefix(raw_text)

        self.log(f"Агрегированный текст (сырой): '{raw_text}'")
        self.log(f"Агрегированный текст (после обрезки): '{text_proc}'")

        if not text_proc:
            result = ClassificationResult(4, "empty aggregated text", 0.0, 0.0, 1.0)
            display_text = "<<шум>>"
            try:
                self.gui_queue.put_nowait((display_text, result))
                self._last_emitted_text = ""
                self.log("Результат (пустой текст) отправлен в GUI")
            except queue.Full:
                self.log("GUI queue full")
            return

        # Максимальная уверенность из всех полученных текстов
        confidence = max(bucket["confidences"]) if bucket["confidences"] else None

        try:
            result = classify_fragment(
                text_proc,
                audio.samples if audio else None,
                audio.sample_rate if audio else config.OUT_RATE,
                confidence=confidence
            )
            self.log(f"Классификация завершена: категория {result.category} ({result.reason})")
        except Exception as e:
            self.log(f"Ошибка классификации: {e}")
            import traceback
            self.log(traceback.format_exc())
            result = ClassificationResult(4, f"classification error: {e}", 0.0, 0.0, 1.0)
            display_text = "<<шум>>"
            try:
                self.gui_queue.put_nowait((display_text, result))
                self._last_emitted_text = ""
                self.log("Результат (ошибка) отправлен в GUI")
            except queue.Full:
                self.log("GUI queue full")
            return

        copy_if_allowed(text_proc, result.category)
        if result.category in (3, 4) or (config.ENABLE_KEYPRESS_ACTIONS and result.category != 1):
            press_enter = True if result.category in (3, 4) else config.AUTOMATION_PRESS_ENTER
            press_category(result.category, press_enter=press_enter, focus_title=config.SITE_WINDOW_TITLE)
        display_text = text_proc if result.category == 1 else _non_russian_placeholder(result.category)
        try:
            self.gui_queue.put_nowait((display_text or "<<empty>>", result))
            self._last_emitted_text = text_proc
            self.log("Результат отправлен в GUI")
        except queue.Full:
            self.log("GUI queue full")

    @staticmethod
    def _add_unique_text(items: list[str], text: str) -> None:
        normalized = text.strip()
        if normalized == "" and "" not in items:
            items.append("")
            return
        if not normalized:
            return
        if any(normalized == item or normalized in item for item in items):
            return
        items[:] = [item for item in items if item not in normalized]
        items.append(normalized)

    def _trim_repeated_prefix(self, text: str) -> str:
        return trim_repeated_prefix(text, self._last_emitted_text, config.MAX_PREFIX_DEDUP_WORDS)


def extract_text(payload: dict[str, Any]) -> str | None:
    for key in ("text", "transcript", "result", "final"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def extract_confidence(payload: dict[str, Any]) -> float | None:
    """Извлекает оценку уверенности из ответа сервера."""
    for key in ("confidence", "score", "confidence_score", "prob"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def extract_timestamp(payload: dict[str, Any]) -> float:
    for key in ("timestamp", "ts", "time", "created_at"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return time.time()


def copy_if_allowed(text: str, category: int) -> None:
    if category != 1 or not text.strip():
        return
    try:
        import pyperclip
        pyperclip.copy(text)
    except Exception:
        pass


async def consumer(ws, gui_queue: queue.Queue, audio_store: AudioFragmentStore, aggregator: TranscriptAggregator, log=print) -> None:
    log("Consumer запущен, ожидание сообщений от сервера")
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                continue
            try:
                parsed = json.loads(msg)
            except Exception:
                log("Failed to parse JSON from server")
                continue
            text = extract_text(parsed)
            if text is None:
                if parsed.get("type") == "ready":
                    continue
                log("Server JSON has no text field, treating as empty transcription")
                text = ""
            timestamp = extract_timestamp(parsed)
            log(f"Получена частичная транскрипция: '{text}' (timestamp {timestamp})")
            audio = audio_store.nearest(timestamp)
            fragment_key = audio.timestamp if audio else _timestamp_bucket(timestamp)
            postprocessor = TranscriptPostprocessor(PostprocessRules(
                config.REPLACEMENTS, config.ASR_REPLACEMENTS,
                config.SPOKEN_NAME_FORMS, config.COLLOQUIAL_REPLACEMENTS,
                config.YO_WORD_REPLACEMENTS
            ))
            text_proc = postprocessor.process(text)
            confidence = extract_confidence(parsed)
            aggregator.submit(fragment_key, text_proc, audio, confidence)
    finally:
        await aggregator.flush_all()
        log("Consumer завершён")


def _timestamp_bucket(timestamp: float) -> float:
    return timestamp - (timestamp % config.CHUNK_SECONDS)


def _non_russian_placeholder(category: int) -> str:
    return {
        2: "<<foreign speech: transcript hidden>>",
        3: "<<unintelligible speech>>",
        4: "<<noise>>",
    }.get(category, "<<unknown>>")