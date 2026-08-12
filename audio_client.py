"""Audio producer and network client loop."""
from __future__ import annotations

import asyncio
import io
import json
import wave
import time

import aiohttp
import numpy as np
import sounddevice as sd
import websockets

import config
from audio_utils import AudioFragmentStore, compute_rms, float32_to_int16_bytes, resample_audio, stereo_to_mono
from consumer import TranscriptAggregator, _timestamp_bucket, consumer


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


async def producer(ws, device_index, audio_store: AudioFragmentStore, start_event, finish_event, aggregator, log=print) -> None:
    """
    start_event  – устанавливается при начале воспроизведения (PLAY->PAUSE)
    finish_event – устанавливается при окончании воспроизведения (PAUSE->PLAY)
    aggregator   – экземпляр TranscriptAggregator для управления таймаутами
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[np.ndarray] = asyncio.Queue()
    chosen = choose_device(device_index)
    use_rate = config.IN_RATE
    if chosen is not None:
        try:
            use_rate = int(sd.query_devices(chosen).get("default_samplerate", config.IN_RATE))
        except Exception:
            use_rate = config.IN_RATE
    log(f"Используется устройство ввода: {chosen} (частота {use_rate} Гц)")

    def callback(indata, frames, time_info, status):
        if status:
            log("sounddevice status: " + str(status))
        loop.call_soon_threadsafe(q.put_nowait, indata.copy())

    stream = sd.InputStream(device=chosen, samplerate=use_rate, channels=config.IN_CHANNELS, dtype="float32", callback=callback)
    semaphore = asyncio.Semaphore(config.UPLOAD_CONCURRENCY)

    audio_buffer = np.array([], dtype=np.float32)
    recording = False
    max_samples = int(config.OUT_RATE * getattr(config, 'MAX_RECORDING_SECONDS', 300))

    stream.start()
    log("Поток захвата аудио запущен")
    try:
        while True:
            mono = stereo_to_mono(await q.get())
            resampled = resample_audio(mono, use_rate, config.OUT_RATE)

            if start_event.is_set():
                start_event.clear()
                if recording:
                    log("Принудительное завершение предыдущей записи (новый START)")
                    if audio_buffer.size > 0:
                        duration = audio_buffer.size / config.OUT_RATE
                        log(f"Отправка накопленного буфера ({duration:.2f} сек)")
                        # Добавляем в хранилище и получаем фрагмент с timestamp
                        fragment = audio_store.add(audio_buffer, config.OUT_RATE)
                        pcm = float32_to_int16_bytes(audio_buffer)
                        for offset in range(0, len(pcm), config.SERVER_MAX_PAYLOAD):
                            await ws.send(pcm[offset : offset + config.SERVER_MAX_PAYLOAD])
                        asyncio.create_task(post_combined_via_rest(pcm, config.OUT_RATE, semaphore, log))
                        # Используем timestamp из фрагмента как ключ
                        aggregator.start_waiting(fragment.timestamp, audio_buffer)
                    recording = False
                    audio_buffer = np.array([], dtype=np.float32)
                recording = True
                audio_buffer = np.array([], dtype=np.float32)
                log("Начало записи аудио (Play->Pause)")

            if finish_event.is_set():
                finish_event.clear()
                if recording and audio_buffer.size > 0:
                    duration = audio_buffer.size / config.OUT_RATE
                    log(f"Окончание записи, отправка {duration:.2f} сек аудио")
                    fragment = audio_store.add(audio_buffer, config.OUT_RATE)
                    pcm = float32_to_int16_bytes(audio_buffer)
                    log(f"Размер PCM: {len(pcm)} байт, отправка по WebSocket")
                    for offset in range(0, len(pcm), config.SERVER_MAX_PAYLOAD):
                        await ws.send(pcm[offset : offset + config.SERVER_MAX_PAYLOAD])
                    log("WebSocket отправка завершена, дублируем через REST")
                    asyncio.create_task(post_combined_via_rest(pcm, config.OUT_RATE, semaphore, log))
                    aggregator.start_waiting(fragment.timestamp, audio_buffer)
                else:
                    log("Запись не велась или буфер пуст – пропускаем отправку")
                recording = False
                audio_buffer = np.array([], dtype=np.float32)
                continue

            if recording:
                audio_buffer = np.concatenate([audio_buffer, resampled])
                if audio_buffer.size > max_samples:
                    log(f"Достигнут лимит буфера ({config.MAX_RECORDING_SECONDS} сек), обрезаем")
                    audio_buffer = audio_buffer[-max_samples:]
    except Exception as e:
        log(f"Ошибка в producer: {e}")
    finally:
        stream.stop()
        stream.close()
        log("Поток захвата аудио остановлен")


async def client_loop(device_index, gui_queue, audio_store: AudioFragmentStore, start_event, finish_event, log=print) -> None:
    aggregator = TranscriptAggregator(gui_queue, log)
    while True:
        try:
            log("Подключение к WebSocket серверу...")
            async with websockets.connect(config.WS_URI, max_size=None) as ws:
                await ws.send(json.dumps({"type": "start", "sample_rate": config.OUT_RATE, "channels": config.OUT_CHANNELS, "format": "pcm16"}))
                log("WebSocket соединение установлено, ждём аудио...")
                await asyncio.gather(
                    producer(ws, device_index, audio_store, start_event, finish_event, aggregator, log),
                    consumer(ws, gui_queue, audio_store, aggregator, log)
                )
        except Exception as exc:
            log("Connection error: " + repr(exc))
            await asyncio.sleep(config.RETRY_DELAY)