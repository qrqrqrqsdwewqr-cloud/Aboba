import numpy as np

from audio_utils import compute_rms, compute_snr_db, spectral_entropy
from classifier import classify_fragment

SR = 48_000

def sine(freq, amp=0.02, seconds=1.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_russian_category():
    assert classify_fragment("Привет как дела").category == 1


def test_foreign_category():
    assert classify_fragment("hello world").category == 2


def test_unintelligible_category():
    assert classify_fragment("ээ ну ...").category == 3


def test_noise_empty_and_symbols():
    assert classify_fragment("").category == 4
    assert classify_fragment("--- ??? 123").category == 4


def test_silence_audio_noise():
    silence = np.zeros(SR, dtype=np.float32)
    result = classify_fragment("", silence, SR)
    assert result.category == 4
    assert compute_rms(silence) == 0.0


def test_cable_hum_audio_metrics():
    hum = sine(50, amp=0.01)
    result = classify_fragment("", hum, SR)
    assert result.category == 4
    assert spectral_entropy(hum) < 0.25


def test_street_noise_audio_metrics():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.02, SR).astype(np.float32)
    result = classify_fragment("", noise, SR)
    assert result.category == 4
    assert spectral_entropy(noise) > 0.8


def test_clean_speech_like_russian_prefers_text():
    speech_like = sine(220, amp=0.02) + sine(440, amp=0.01)
    assert classify_fragment("это тестовая фраза", speech_like, SR).category == 1


def test_mumbling_is_unintelligible():
    mumble = sine(120, amp=0.005)
    assert classify_fragment("мм эээ", mumble, SR).category == 3


def test_snr_estimator_runs():
    x = np.concatenate([np.zeros(SR // 2), sine(300, seconds=0.5)])
    assert compute_snr_db(x) >= 0.0
