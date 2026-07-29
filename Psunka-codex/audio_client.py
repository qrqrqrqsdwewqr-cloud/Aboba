"""Audio capture and local STT client loop."""
from __future__ import annotations

import asyncio, time
from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

import config
from audio_utils import AudioFragmentStore, compute_rms, resample_audio, stereo_to_mono
from classifier import classify_fragment
from gigaam_client import GigaAMClient
from language_detector import detect_language
from postprocessing import PostprocessRules, TranscriptPostprocessor
from clipboard_utils import copy_transcript

@dataclass(frozen=True)
class FullAudioJob:
    samples: np.ndarray
    sample_rate: int
    timestamp: float

def list_devices() -> None:
    print("Available audio devices:")
    for index, dev in enumerate(sd.query_devices()):
        print(f"{index}: {dev['name']} (max_in:{dev['max_input_channels']}, max_out:{dev['max_output_channels']}, default_sr:{dev.get('default_samplerate')})")

def choose_device(device_index):
    if device_index is not None: return device_index
    hints = ("cable", "vb-audio", "stereo mix", "loopback", "virtual")
    for index, dev in enumerate(sd.query_devices()):
        if any(h in dev["name"].lower() for h in hints):
            print(f"Auto-selected device {index}: {dev['name']}"); return index
    print("No matching virtual cable found; using default input device"); return None

async def client_loop(device_index, gui_queue, audio_store: AudioFragmentStore, recording_event=None, finish_event=None, stop_event=None, log=print) -> None:
    if config.STT_BACKEND != "gigaam":
        raise RuntimeError("Legacy WebSocket/REST STT backend is disabled from the main path; set STT_BACKEND='gigaam'.")
    producer_q: asyncio.Queue[np.ndarray] = asyncio.Queue(); jobs: asyncio.Queue[FullAudioJob] = asyncio.Queue(maxsize=1)
    client = GigaAMClient(); post = TranscriptPostprocessor(PostprocessRules(config.REPLACEMENTS, config.ASR_REPLACEMENTS, config.SPOKEN_NAME_FORMS, config.COLLOQUIAL_REPLACEMENTS, config.YO_WORD_REPLACEMENTS))
    chosen = choose_device(device_index); use_rate = _device_rate(chosen); loop = asyncio.get_running_loop()
    def callback(indata, frames, time_info, status):
        if status: log("sounddevice status: " + str(status))
        loop.call_soon_threadsafe(producer_q.put_nowait, indata.copy())
    stream = sd.InputStream(device=chosen, samplerate=use_rate, channels=config.IN_CHANNELS, dtype="float32", callback=callback)
    stream.start()
    try:
        await asyncio.gather(_capture_full_jobs(producer_q, jobs, audio_store, use_rate, recording_event, finish_event, stop_event, log), _transcribe_jobs(jobs, gui_queue, client, post, stop_event, log))
    finally:
        stream.stop(); stream.close(); client.close()

async def _capture_full_jobs(q, jobs, audio_store, use_rate, recording_event, finish_event, stop_event, log):
    prefix_samples = int(use_rate * config.CAPTURE_PREFIX_SECONDS); suffix_samples = int(use_rate * config.CAPTURE_SUFFIX_SECONDS)
    prefix = deque(); prefix_len = 0; recording = False; chunks = []
    while stop_event is None or not stop_event.is_set():
        mono = stereo_to_mono(await q.get())
        if not recording:
            prefix.append(mono); prefix_len += mono.size
            while prefix_len > prefix_samples and prefix:
                removed = prefix.popleft(); prefix_len -= removed.size
            if recording_event is not None and recording_event.is_set():
                recording = True; chunks = list(prefix); recording_event.clear(); log("Начата запись полного задания")
            continue
        chunks.append(mono)
        if finish_event is not None and finish_event.is_set():
            finish_event.clear()
            remaining = suffix_samples
            while remaining > 0:
                extra = stereo_to_mono(await q.get()); chunks.append(extra); remaining -= extra.size
            samples = np.concatenate(chunks).astype(np.float32) if chunks else np.array([], dtype=np.float32)
            fragment = audio_store.add(samples, use_rate)
            if config.PROCESS_ONLY_ON_VOICE and compute_rms(samples) <= config.RMS_SILENCE_THRESHOLD:
                recording = False; chunks = []; continue
            await jobs.put(FullAudioJob(fragment.samples, fragment.sample_rate, fragment.timestamp))
            recording = False; chunks = []; log("Завершена запись полного задания")

async def _transcribe_jobs(jobs, gui_queue, client: GigaAMClient, post, stop_event, log):
    while stop_event is None or not stop_event.is_set():
        job = await jobs.get()
        result = await asyncio.to_thread(client.transcribe, job.samples, job.sample_rate)
        text = post.process(result.text)
        lang = detect_language(text)
        classification = classify_fragment(text, job.samples, job.sample_rate, language=lang.language)
        if classification.category in (config.CATEGORY_RUSSIAN,): copy_transcript(text)
        display = text if classification.category in (1, 2) else _placeholder(classification.category)
        try: gui_queue.put_nowait((display or "<<empty>>", classification, text, lang, result))
        except Exception: log("GUI queue full")

def _device_rate(chosen) -> int:
    if chosen is not None:
        try: return int(sd.query_devices(chosen).get("default_samplerate", config.IN_RATE))
        except Exception: pass
    return config.IN_RATE

def _placeholder(category: int) -> str:
    return {2:"<<foreign speech: transcript hidden>>", 3:"<<unintelligible speech>>", 4:"<<noise>>"}.get(category, "<<unknown>>")
