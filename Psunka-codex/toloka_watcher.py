"""Toloka player state watcher and named ROI storage."""
from __future__ import annotations

import json, logging, threading, time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import config
import numpy as np

LOG = logging.getLogger(__name__)

class TolokaState(str, Enum):
    LOADING = "LOADING"
    PLAY_ICON = "PLAY_ICON"
    PAUSE_ICON = "PAUSE_ICON"
    UNKNOWN = "UNKNOWN"

BUSY_STATES = {"RECORDING", "TRANSCRIBING", "CLASSIFYING", "SELECTING_CATEGORY", "SUBMITTING_CATEGORY", "WAITING_FOR_TEXT_FIELD", "PASTING_TEXT", "SUBMITTING_TEXT"}

@dataclass(frozen=True)
class SearchRegion:
    left: int; top: int; width: int; height: int
    def as_mss(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)

NAMED_REGION_KEYS = {"player", "category_panel", "submit_button", "text_field"}

def _read_config() -> dict:
    path = Path(config.TOLOKA_CONFIG_PATH)
    if not path.exists(): return {}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("Не удалось прочитать config.json: %r", exc); return {}

def _write_config(data: dict) -> None:
    Path(config.TOLOKA_CONFIG_PATH).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _region_from_dict(value: dict | None) -> SearchRegion | None:
    try:
        if not value: return None
        region = SearchRegion(int(value["left"]), int(value["top"]), int(value["width"]), int(value["height"]))
        return region if region.width > 0 and region.height > 0 else None
    except Exception:
        return None

def load_region(name: str = "player") -> SearchRegion | None:
    data = _read_config()
    if name in {"category_1", "category_2", "category_3", "category_4"}:
        return _region_from_dict((data.get("category_checkbox_regions") or {}).get(name[-1]))
    region = _region_from_dict((data.get("regions") or {}).get(name))
    if region is None and name == "player":
        region = _region_from_dict(data.get("toloka_search_region"))
    return region

def save_region(name: str, region: SearchRegion) -> None:
    data = _read_config()
    if name in {"category_1", "category_2", "category_3", "category_4"}:
        data.setdefault("category_checkbox_regions", {})[name[-1]] = region.__dict__
    else:
        data.setdefault("regions", {})[name] = region.__dict__
    if name == "player":
        data.pop("toloka_search_region", None)
    _write_config(data)
    LOG.info("Сохранена область %s: %s", name, region)

def select_region_interactively(name: str = "player") -> SearchRegion | None:
    try:
        import tkinter as tk
    except Exception as exc:
        LOG.error("Выбор области недоступен: %r", exc); return None
    result: dict[str, int] = {}
    root = tk.Tk(); root.attributes("-fullscreen", True); root.attributes("-alpha", 0.25); root.attributes("-topmost", True); root.configure(bg="black")
    canvas = tk.Canvas(root, cursor="cross", bg="black", highlightthickness=0); canvas.pack(fill="both", expand=True)
    start = {"x": 0, "y": 0, "rect": None}
    def on_down(event):
        start.update(x=event.x_root, y=event.y_root); start["rect"] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=3)
    def on_move(event):
        if start["rect"] is not None: canvas.coords(start["rect"], start["x"], start["y"], event.x_root, event.y_root)
    def on_up(event):
        x1, y1, x2, y2 = start["x"], start["y"], event.x_root, event.y_root
        result.update(left=min(x1, x2), top=min(y1, y2), width=abs(x2-x1), height=abs(y2-y1)); root.destroy()
    canvas.bind("<ButtonPress-1>", on_down); canvas.bind("<B1-Motion>", on_move); canvas.bind("<ButtonRelease-1>", on_up); root.mainloop()
    if result.get("width", 0) < 5 or result.get("height", 0) < 5: return None
    region = SearchRegion(**result); save_region(name, region); return region

class TolokaWatcher:
    def __init__(self, on_playback_started: Callable[[], None] | None = None, on_playback_finished: Callable[[], None] | None = None, log=LOG.info, state_provider: Callable[[], str] | None = None):
        self.on_playback_started = on_playback_started or (lambda: None)
        self.on_playback_finished = on_playback_finished or (lambda: None)
        self.log = log; self.state_provider = state_provider or (lambda: "IDLE")
        self._stop = threading.Event(); self._thread: threading.Thread | None = None
        self._candidate_state = TolokaState.UNKNOWN; self._candidate_count = 0; self._confirmed_state = TolokaState.UNKNOWN
        self._templates = None

    def start(self, *, sync: bool = True) -> bool:
        if self._thread and self._thread.is_alive(): return True
        self._stop.clear(); region = load_region("player") or select_region_interactively("player")
        if region is None: self.log("Watcher не запущен: область player не выбрана"); return False
        try: self._load_templates()
        except Exception as exc: self.log(f"Watcher не запущен: {exc!r}"); return False
        if sync:
            self._candidate_state = self._confirmed_state = TolokaState.UNKNOWN; self._candidate_count = 0
        self._thread = threading.Thread(target=self._loop, args=(region,), daemon=True); self._thread.start(); return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive(): self._thread.join(timeout=1.5)
        self._thread = None

    def _load_templates(self) -> None:
        if self._templates is not None: return
        import cv2
        base = Path(__file__).resolve().parent / "templates"; templates = []
        for state, filename in ((TolokaState.LOADING,"loading.png"),(TolokaState.PLAY_ICON,"play.png"),(TolokaState.PAUSE_ICON,"pause.png")):
            image = cv2.imread(str(base/filename), cv2.IMREAD_GRAYSCALE)
            if image is None: raise FileNotFoundError(base/filename)
            templates.append((state, image))
        self._templates = templates

    def _loop(self, region: SearchRegion) -> None:
        import cv2, mss
        with mss.mss() as sct:
            while not self._stop.is_set():
                shot = sct.grab(region.as_mss()); frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2GRAY)
                self._handle_detected_state(self._detect(frame, cv2)); self._stop.wait(config.TOLOKA_WATCH_INTERVAL_SECONDS)

    def _handle_detected_state(self, state: TolokaState) -> None:
        if state != self._candidate_state:
            self._candidate_state = state; self._candidate_count = 1; return
        self._candidate_count += 1
        if self._candidate_count >= config.WATCHER_STABLE_DETECTIONS and state != self._confirmed_state:
            previous = self._confirmed_state; self._confirmed_state = state; self._handle_confirmed_transition(previous, state)

    def _handle_confirmed_transition(self, previous: TolokaState, current: TolokaState) -> None:
        if self.state_provider() in BUSY_STATES: return
        if previous == TolokaState.PLAY_ICON and current == TolokaState.PAUSE_ICON: self.on_playback_started()
        elif previous == TolokaState.PAUSE_ICON and current == TolokaState.PLAY_ICON: self.on_playback_finished()

    def _detect(self, frame, cv2) -> TolokaState:
        best = (TolokaState.UNKNOWN, 0.0)
        for state, template in self._templates or []:
            if frame.shape[0] < template.shape[0] or frame.shape[1] < template.shape[1]: continue
            score = float(cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED).max())
            if score > best[1]: best = (state, score)
        return best[0] if best[1] >= config.TOLOKA_TEMPLATE_THRESHOLD else TolokaState.UNKNOWN
