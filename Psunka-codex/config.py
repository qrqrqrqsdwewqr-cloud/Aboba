"""Application configuration for speech classification client."""
from __future__ import annotations

DEVICE = None
IN_RATE = 48_000
IN_CHANNELS = 1
CHUNK_SECONDS = 7
COMBINE_COUNT = 1
TRANSCRIPT_FLUSH_DELAY_SECONDS = 0.2
STT_CONTEXT_PREFIX_SECONDS = 0.2
STT_CONTEXT_SUFFIX_SECONDS = 0.2
MAX_PREFIX_DEDUP_WORDS = 2
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
SHORT_FRAGMENT_MAX_CHARS = 3
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
YO_WORD_REPLACEMENTS = {
    "еще": "ещё",
    "все": "всё",
    "ее": "её",
    "елка": "ёлка",
    "елки": "ёлки",
    "ежик": "ёжик",
    "ежики": "ёжики",
    "еж": "ёж",
    "ежи": "ежи",
    "береза": "берёза",
    "березы": "берёзы",
    "легкий": "лёгкий",
    "легкая": "лёгкая",
    "легкое": "лёгкое",
    "легкие": "лёгкие",
    "самолет": "самолёт",
    "самолеты": "самолёты",
}

# Optional GUI-controlled keyboard/site automation.
ENABLE_KEYPRESS_ACTIONS = False
AUTO_SEND_RUSSIAN_TO_SITE = False
SITE_WINDOW_TITLE = None
AUTOMATION_PRESS_ENTER = False
AUTOMATION_KEY_DELAY_SECONDS = 0.05
SITE_PRE_OPEN_DELAY_SECONDS = 0.25
SITE_AFTER_TYPE_DELAY_SECONDS = 0.12
SITE_SUBMIT_AFTER_PASTE = False
USE_CLIPBOARD_PASTE = True
TEXT_FIELD_READY_TIMEOUT_SECONDS = 2.0
TEXT_FIELD_READY_CHECK_INTERVAL_SECONDS = 0.2

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
}
