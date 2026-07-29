"""Audio producer and network client loop."""
from __future__ import annotations

import asyncio
import io
import json
import wave

import aiohttp
import numpy as np
import sounddevice as sd
import websockets

import config
from audio_utils import AudioFragmentStore, compute_rms, float32_to_int16_bytes, resample_audio, stereo_to_mono
from consumer import consumer


def list_devices() -> None:
    print("Available audio devices:")
    for index, dev in enumerate(sd.query_devices()):
        print(f"{index}: {dev['name']} (max_in:{dev['max_input_channels']}, max_out:{dev['max_output_channels']}, default_sr:{dev.get('default_samplerate')})")


def choose_device(device_index):
    if device_index is not None:
        return device_index
    hints = ("cable", "vb-audio", "stereo mix", "loopback", "virtual")
    for index, dev in enumerate(sd.query_devices()):
        if any(hint in dev["name"].lower() for hint in hints):
            print(f"Auto-selected device {index}: {dev['name']}")
            return index
    print("No matching virtual cable found; using default input device")
    return None


async def post_combined_via_rest(pcm_bytes: bytes, sample_rate: int, semaphore: asyncio.Semaphore, log=print) -> None:
    async with semaphore:
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_bytes)
            data = wav_io.getvalue()
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            form = aiohttp.FormData()
            form.add_field("file", data, filename="combined.wav", content_type="audio/wav")
            async with session.post(config.REST_URI, data=form) as resp:
                log(f"REST upload status: {resp.status}, response: {await resp.text()}")


async def producer(ws, device_index, audio_store: AudioFragmentStore, process_event=None, log=print) -> None:
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[np.ndarray] = asyncio.Queue()
    chosen = choose_device(device_index)
    use_rate = config.IN_RATE
    if chosen is not None:
        try:
            use_rate = int(sd.query_devices(chosen).get("default_samplerate", config.IN_RATE))
        except Exception:
            use_rate = config.IN_RATE

    def callback(indata, frames, time_info, status):
        if status:
            log("sounddevice status: " + str(status))
        loop.call_soon_threadsafe(q.put_nowait, indata.copy())

    stream = sd.InputStream(device=chosen, samplerate=use_rate, channels=config.IN_CHANNELS, dtype="float32", callback=callback)
    semaphore = asyncio.Semaphore(config.UPLOAD_CONCURRENCY)
    frag_samples = int(config.OUT_RATE * config.CHUNK_SECONDS)
    suffix_samples = int(config.OUT_RATE * config.STT_CONTEXT_SUFFIX_SECONDS)
    prefix_samples = int(config.OUT_RATE * config.STT_CONTEXT_PREFIX_SECONDS)
    previous_tail = np.array([], dtype=np.float32)
    chunks: list[np.ndarray] = []
    length = 0
    combine: list[bytes] = []
    stream.start()
    try:
        while True:
            mono = stereo_to_mono(await q.get())
            resampled = resample_audio(mono, use_rate, config.OUT_RATE)
            chunks.append(resampled)
            length += resampled.size
            while length >= frag_samples + suffix_samples:
                needed = frag_samples
                parts: list[np.ndarray] = []
                while needed:
                    chunk = chunks.pop(0)
                    parts.append(chunk[:needed])
                    if chunk.size > needed:
                        chunks.insert(0, chunk[needed:])
                    needed -= min(needed, chunk.size)
                length -= frag_samples
                core_fragment = np.concatenate(parts).astype(np.float32)
                suffix_fragment = _peek_samples(chunks, suffix_samples)
                stt_fragment = np.concatenate([previous_tail, core_fragment, suffix_fragment]).astype(np.float32)
                previous_tail = core_fragment[-prefix_samples:].copy() if prefix_samples else np.array([], dtype=np.float32)
                audio_store.add(stt_fragment, config.OUT_RATE)
                if config.PROCESS_ONLY_ON_VOICE and compute_rms(core_fragment) <= config.RMS_SILENCE_THRESHOLD:
                    continue
                if process_event is not None and not process_event.is_set():
                    continue
                if process_event is not None:
                    process_event.clear()
                pcm = float32_to_int16_bytes(stt_fragment)
                for offset in range(0, len(pcm), config.SERVER_MAX_PAYLOAD):
                    await ws.send(pcm[offset : offset + config.SERVER_MAX_PAYLOAD])
                combine.append(pcm)
                if len(combine) >= config.COMBINE_COUNT:
                    asyncio.create_task(post_combined_via_rest(b"".join(combine), config.OUT_RATE, semaphore, log))
                    combine.clear()
    finally:
        stream.stop(); stream.close()


def _peek_samples(chunks: list[np.ndarray], sample_count: int) -> np.ndarray:
    if sample_count <= 0:
        return np.array([], dtype=np.float32)
    parts: list[np.ndarray] = []
    remaining = sample_count
    for chunk in chunks:
        if remaining <= 0:
            break
        take = min(remaining, chunk.size)
        parts.append(chunk[:take])
        remaining -= take
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


async def client_loop(device_index, gui_queue, audio_store: AudioFragmentStore, process_event=None, log=print) -> None:
    while True:
        try:
            async with websockets.connect(config.WS_URI, max_size=None) as ws:
                await ws.send(json.dumps({"type": "start", "sample_rate": config.OUT_RATE, "channels": config.OUT_CHANNELS, "format": "pcm16"}))
                await asyncio.gather(producer(ws, device_index, audio_store, process_event, log), consumer(ws, gui_queue, audio_store, log))
        except Exception as exc:
            log("Connection error: " + repr(exc))
            await asyncio.sleep(config.RETRY_DELAY)
