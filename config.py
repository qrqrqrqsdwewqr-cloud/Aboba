"""Application configuration for speech classification client."""
from __future__ import annotations

import yo_words  # <-- добавлен импорт библиотеки слов с "ё"

DEVICE = None
IN_RATE = 48_000
IN_CHANNELS = 1
CHUNK_SECONDS = 7
COMBINE_COUNT = 1
TRANSCRIPT_FLUSH_DELAY_SECONDS = 0.8
STT_CONTEXT_PREFIX_SECONDS = 0.2
STT_CONTEXT_SUFFIX_SECONDS = 0.2
MAX_PREFIX_DEDUP_WORDS = 5
WS_URI = "ws://127.0.0.1:9876/v1/ws"
REST_URI = "http://127.0.0.1:9876/v1/transcribe"
OUT_RATE = 48_000
OUT_CHANNELS = 1
RETRY_DELAY = 0.5
SERVER_MAX_PAYLOAD = 524_288
UPLOAD_CONCURRENCY = 1
GUI_POLL_INTERVAL_MS = 100
PROCESS_ONLY_ON_VOICE = False
PROCESS_ONLY_ON_TOLOKA_PLAY_PAUSE = True
TOLOKA_WATCH_INTERVAL_SECONDS = 0.1
TOLOKA_TEMPLATE_THRESHOLD = 0.82
TOLOKA_CONFIG_PATH = "config.json"

# Timestamp matching
AUDIO_MATCH_WINDOW_SECONDS = 10.0
AUDIO_STORE_MAX_FRAGMENTS = 80

# Audio/noise thresholds
RMS_SILENCE_THRESHOLD = 0.0006
RMS_MUMBLE_THRESHOLD = 0.004
SNR_NOISE_THRESHOLD_DB = 6.0
SNR_UNINTELLIGIBLE_THRESHOLD_DB = 9.0
SPECTRAL_ENTROPY_NOISE_THRESHOLD = 0.82
SPECTRAL_ENTROPY_TONE_THRESHOLD = 0.22
NOISE_FLOOR_PERCENTILE = 10
DEGRADE_RUSSIAN_ON_BAD_AUDIO = True

# Optional filters
ENABLE_BANDPASS_FILTER = False
BANDPASS_LOW_HZ = 80.0
BANDPASS_HIGH_HZ = 7_500.0
ENABLE_NOTCH_FILTER = False
NOTCH_FREQ_HZ = 50.0
NOTCH_Q = 30.0

# Classification text heuristics
MIN_RUSSIAN_WORD_LEN = 2
SHORT_FRAGMENT_MAX_CHARS = 5
GARBAGE_SYMBOL_RATIO = 0.65
FRAGMENT_MARKERS = {"а", "э", "ээ", "эм", "мм", "м", "ну", "ау", "о", "ой"}
LAUGHTER_MARKERS = {"ха", "хаха", "ахаха", "ха-ха"}
ASR_REPLACEMENTS = {}
SPOKEN_NAME_FORMS = {"саня": "Саша", "саш": "Саша", "дима": "Дмитрий"}
COLLOQUIAL_REPLACEMENTS = {"чё": "что", "че": "что", "щас": "сейчас", "тыща": "тысяча"}
CATEGORY_KEYS = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    "russian": "1",
    "music": "1",
    "speech": "2",
    "foreign": "2",
    "noise": "3",
    "unintelligible": "3",
    "other": "4",
}
REPLACEMENTS = {"чё": "что", "че": "что", "щас": "сейчас", "тыща": "тысяча"}

# ---- АВТОМАТИЧЕСКАЯ ГЕНЕРАЦИЯ СЛОВАРЯ ЗАМЕН ДЛЯ БУКВЫ "Ё" ----
def _build_yo_replacements():
    replacements = {}
    for word in yo_words.YO_WORDS:
        key = word.replace("ё", "е")
        if key != word:
            replacements[key] = word
    return replacements

YO_WORD_REPLACEMENTS = _build_yo_replacements()

# Optional GUI-controlled keyboard/site automation.
ENABLE_KEYPRESS_ACTIONS = False
AUTO_SEND_RUSSIAN_TO_SITE = False
SITE_WINDOW_TITLE = "Microsoft Edge"
AUTOMATION_PRESS_ENTER = False
AUTOMATION_KEY_DELAY_SECONDS = 0.05
SITE_PRE_OPEN_DELAY_SECONDS = 0.25
SITE_AFTER_TYPE_DELAY_SECONDS = 0.12
SITE_SUBMIT_AFTER_PASTE = False
USE_CLIPBOARD_PASTE = True
TEXT_FIELD_READY_TIMEOUT_SECONDS = 2.0
TEXT_FIELD_READY_CHECK_INTERVAL_SECONDS = 0.2

MAX_RECORDING_SECONDS = 300

# ---------- НАСТРОЙКИ ДЛЯ КЛИКОВ ПО КАТЕГОРИЯМ ----------
ENABLE_CATEGORY_CLICK = True
CATEGORIES_TO_CLICK = [1, 3, 4]
CATEGORY_REGION_KEYS = {
    1: "category_1_region",
    3: "category_3_region",
    4: "category_4_region",
}
CATEGORY_CLICK_DELAY = 0.3

# ---------- КЛЮЧИ ДЛЯ ОБЛАСТЕЙ КНОПКИ "ОТПРАВИТЬ" ----------
SEND_REGION_KEY = "send_button_region"
SEND_REGION_2_KEY = "send_button_region_2"

# ---------- ЗАДЕРЖКА ПЕРЕД ВСТАВКОЙ (CTRL+V) ПОСЛЕ КЛИКА ----------
PASTE_DELAY_SECONDS = 1.3

# ---------- НАСТРОЙКИ КЛАССИФИКАЦИИ ТЕКСТА ----------
SHORT_TEXT_MAX_LEN = 3

# ---------- НАСТРОЙКИ ДЛЯ КОРОТКИХ СЛОВ ----------
SHORT_WORD_CONFIDENCE_THRESHOLD = 0.6
SHORT_WORD_MIN_SPEECH_DURATION = 0.3
SHORT_WORD_SNR_THRESHOLD = 4.0
SHORT_WORD_SPEECH_RATIO = 0.25

# ---------- НАСТРОЙКИ ДЛЯ ОСОБЫХ КОРОТКИХ СЛОВ (не, да, угу, ага, нет) ----------
SPECIAL_SHORT_WORDS = {"не", "нет", "да", "угу", "ага"}
SPECIAL_SHORT_WORD_CONFIDENCE_THRESHOLD = 0.7
SPECIAL_SHORT_WORD_MIN_SNR = 6.0
SPECIAL_SHORT_WORD_MIN_SPEECH_RATIO = 0.4
SPECIAL_SHORT_WORD_MIN_DURATION = 0.3

# ---------- НАСТРОЙКИ ДЛЯ ГАЛЛЮЦИНАЦИЙ ----------
HALLUCINATION_WORDS = {"угу", "ага", "да", "нет", "э", "м", "ну", "ой", "ах", "ох"}
HALLUCINATION_SNR_THRESHOLD = 10.0
HALLUCINATION_SPEECH_RATIO = 0.5
SHORT_SPEECH_DURATION_THRESHOLD_SEC = 0.8

# ---------- НАСТРОЙКИ VAD ----------
VAD_SAMPLE_RATE = 16000
SILERO_VAD_THRESHOLD = 0.1
SILERO_MIN_SPEECH_DURATION_MS = 250
SILERO_MIN_SILENCE_DURATION_MS = 100
SILERO_SPEECH_RATIO_THRESHOLD = 0.1

WEBRTC_AGRESSIVENESS = 2
WEBRTC_FRAME_DURATION_MS = 30

VAD_SPEECH_RATIO_HIGH = 0.25
VAD_SPEECH_RATIO_LOW = 0.24

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sounddevice": "sounddevice",
    "websockets": "websockets",
    "aiohttp": "aiohttp",
    "PyQt5": "PyQt5",
    "pyperclip": "pyperclip",
    "cv2": "opencv-python",
    "mss": "mss",
    "webrtcvad": "webrtcvad",
    "silero-vad": "silero-vad",
    "torch": "torch",
}