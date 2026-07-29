from toloka_watcher import TolokaState, TolokaWatcher


def test_play_unknown_pause_completes_playback_cycle():
    calls = []
    watcher = TolokaWatcher(lambda: calls.append("done"))

    watcher._handle_playback_state(TolokaState.PLAY)
    watcher._handle_playback_state(TolokaState.UNKNOWN)
    watcher._handle_playback_state(TolokaState.UNKNOWN)
    watcher._handle_playback_state(TolokaState.PAUSE)

    assert calls == ["done"]


def test_pause_without_play_does_not_complete_cycle():
    calls = []
    watcher = TolokaWatcher(lambda: calls.append("done"))

    watcher._handle_playback_state(TolokaState.UNKNOWN)
    watcher._handle_playback_state(TolokaState.PAUSE)

    assert calls == []


def test_loading_resets_play_seen():
    calls = []
    watcher = TolokaWatcher(lambda: calls.append("done"))

    watcher._handle_playback_state(TolokaState.PLAY)
    watcher._handle_playback_state(TolokaState.LOADING)
    watcher._handle_playback_state(TolokaState.PAUSE)

    assert calls == []
