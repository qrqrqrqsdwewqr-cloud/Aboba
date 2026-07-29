"""Toloka player state watcher based only on template matching screenshots."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import config
import numpy as np

LOG = logging.getLogger(__name__)


class TolokaState(str, Enum):
    LOADING = "LOADING"
    PLAY = "PLAY"
    PAUSE = "PAUSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SearchRegion:
    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


def load_region() -> SearchRegion | None:
    path = Path(config.TOLOKA_CONFIG_PATH)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        region = data.get("toloka_search_region") or {}
        return SearchRegion(int(region["left"]), int(region["top"]), int(region["width"]), int(region["height"]))
    except Exception as exc:
        LOG.warning("Не удалось загрузить область Toloka: %r", exc)
        return None


def save_region(region: SearchRegion) -> None:
    path = Path(config.TOLOKA_CONFIG_PATH)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["toloka_search_region"] = region.__dict__
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Выбрана область Toloka: %s", region)


def select_region_interactively() -> SearchRegion | None:
    """Let user drag a screen rectangle using a tiny Tk overlay."""
    try:
        import tkinter as tk
    except Exception as exc:
        LOG.error("Выбор области недоступен: %r", exc)
        return None
    result: dict[str, int] = {}
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.25)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    canvas = tk.Canvas(root, cursor="cross", bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    start = {"x": 0, "y": 0, "rect": None}

    def on_down(event):
        start.update(x=event.x_root, y=event.y_root)
        start["rect"] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=3)

    def on_move(event):
        if start["rect"] is not None:
            canvas.coords(start["rect"], start["x"], start["y"], event.x_root, event.y_root)

    def on_up(event):
        x1, y1, x2, y2 = start["x"], start["y"], event.x_root, event.y_root
        result.update(left=min(x1, x2), top=min(y1, y2), width=abs(x2 - x1), height=abs(y2 - y1))
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_down)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_up)
    root.mainloop()
    if result.get("width", 0) < 5 or result.get("height", 0) < 5:
        return None
    region = SearchRegion(**result)
    save_region(region)
    return region


class TolokaWatcher:
    def __init__(self, on_play_to_pause, log=LOG.info):
        self.on_play_to_pause = on_play_to_pause
        self.log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_state = TolokaState.UNKNOWN
        self._play_seen = False
        self._templates = None

    def start(self, *, sync: bool = True) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        region = load_region()
        if region is None:
            region = select_region_interactively()
        if region is None:
            self.log("Watcher не запущен: область поиска не выбрана")
            return False
        try:
            self._load_templates()
        except Exception as exc:
            self.log(f"Watcher не запущен: {exc!r}")
            return False
        if sync:
            self._last_state = TolokaState.UNKNOWN
            self._play_seen = False
        self._thread = threading.Thread(target=self._loop, args=(region,), daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

    def _load_templates(self) -> None:
        if self._templates is not None:
            return
        import cv2
        templates = []
        base = Path(__file__).resolve().parent / "templates"
        for state, filename in ((TolokaState.LOADING, "loading.png"), (TolokaState.PLAY, "play.png"), (TolokaState.PAUSE, "pause.png")):
            image = cv2.imread(str(base / filename), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(base / filename)
            templates.append((state, image))
        self._templates = templates

    def _loop(self, region: SearchRegion) -> None:
        import cv2
        import mss
        with mss.mss() as sct:
            while not self._stop.is_set():
                shot = sct.grab(region.as_mss())
                frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2GRAY)
                state = self._detect(frame, cv2)
                if state != self._last_state:
                    self._last_state = state
                self._handle_playback_state(state)
                self._stop.wait(config.TOLOKA_WATCH_INTERVAL_SECONDS)

    def _handle_playback_state(self, state: TolokaState) -> None:
        if state == TolokaState.PLAY:
            if not self._play_seen:
                self._play_seen = True
            return
        if state == TolokaState.UNKNOWN:
            return
        if state == TolokaState.PAUSE:
            if self._play_seen:
                self._play_seen = False
                self.on_play_to_pause()
            return
        if state == TolokaState.LOADING:
            self._play_seen = False

    def _detect(self, frame, cv2) -> TolokaState:
        for state, template in self._templates or []:
            if frame.shape[0] < template.shape[0] or frame.shape[1] < template.shape[1]:
                continue
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            if float(result.max()) >= config.TOLOKA_TEMPLATE_THRESHOLD:
                return state
        return TolokaState.UNKNOWN
