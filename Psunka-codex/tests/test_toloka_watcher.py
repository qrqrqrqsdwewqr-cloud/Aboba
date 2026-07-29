import config
from toloka_watcher import TolokaState, TolokaWatcher


def stable(watcher, state):
    for _ in range(config.WATCHER_STABLE_DETECTIONS):
        watcher._handle_detected_state(state)


def test_play_to_pause_starts_once():
    calls = []
    watcher = TolokaWatcher(lambda: calls.append("start"), lambda: calls.append("finish"))
    stable(watcher, TolokaState.PLAY_ICON)
    stable(watcher, TolokaState.PAUSE_ICON)
    stable(watcher, TolokaState.PAUSE_ICON)
    assert calls == ["start"]


def test_pause_to_play_finishes():
    calls = []
    watcher = TolokaWatcher(lambda: calls.append("start"), lambda: calls.append("finish"))
    stable(watcher, TolokaState.PAUSE_ICON)
    stable(watcher, TolokaState.PLAY_ICON)
    assert calls == ["finish"]


def test_busy_state_suppresses_callbacks():
    calls = []
    watcher = TolokaWatcher(lambda: calls.append("start"), lambda: calls.append("finish"), state_provider=lambda: "RECORDING")
    stable(watcher, TolokaState.PLAY_ICON)
    stable(watcher, TolokaState.PAUSE_ICON)
    assert calls == []
