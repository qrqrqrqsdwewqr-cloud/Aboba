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

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)


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


# ---------- ФУНКЦИИ ДЛЯ ОБЛАСТЕЙ КАТЕГОРИЙ ----------
def load_category_region(category: int) -> SearchRegion | None:
    path = Path(config.TOLOKA_CONFIG_PATH)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        key = config.CATEGORY_REGION_KEYS.get(category)
        if not key:
            return None
        region_data = data.get(key)
        if not region_data:
            return None
        return SearchRegion(
            int(region_data["left"]),
            int(region_data["top"]),
            int(region_data["width"]),
            int(region_data["height"])
        )
    except Exception as exc:
        LOG.warning("Не удалось загрузить область для категории %d: %r", category, exc)
        return None


def save_category_region(category: int, region: SearchRegion) -> None:
    path = Path(config.TOLOKA_CONFIG_PATH)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    key = config.CATEGORY_REGION_KEYS.get(category)
    if not key:
        LOG.error("Нет ключа для категории %d", category)
        return
    data[key] = region.__dict__
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Сохранена область для категории %d: %s", category, region)


def select_category_region_interactively(category: int) -> SearchRegion | None:
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
    save_category_region(category, region)
    return region


def select_region_interactively() -> SearchRegion | None:
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


# ---------- ФУНКЦИИ ДЛЯ ОБЛАСТИ КНОПКИ "ОТПРАВИТЬ" ----------
def load_send_region() -> SearchRegion | None:
    path = Path(config.TOLOKA_CONFIG_PATH)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        region_data = data.get(config.SEND_REGION_KEY)
        if not region_data:
            return None
        return SearchRegion(
            int(region_data["left"]),
            int(region_data["top"]),
            int(region_data["width"]),
            int(region_data["height"])
        )
    except Exception as exc:
        LOG.warning("Не удалось загрузить область отправки: %r", exc)
        return None


def save_send_region(region: SearchRegion) -> None:
    path = Path(config.TOLOKA_CONFIG_PATH)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[config.SEND_REGION_KEY] = region.__dict__
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Сохранена область отправки: %s", region)


def select_send_region_interactively() -> SearchRegion | None:
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
    save_send_region(region)
    return region


# ---------- НОВЫЕ ФУНКЦИИ ДЛЯ ОБЛАСТИ КНОПКИ "ОТПРАВИТЬ 2" ----------
def load_send_region_2() -> SearchRegion | None:
    path = Path(config.TOLOKA_CONFIG_PATH)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        region_data = data.get(config.SEND_REGION_2_KEY)
        if not region_data:
            return None
        return SearchRegion(
            int(region_data["left"]),
            int(region_data["top"]),
            int(region_data["width"]),
            int(region_data["height"])
        )
    except Exception as exc:
        LOG.warning("Не удалось загрузить область отправки 2: %r", exc)
        return None


def save_send_region_2(region: SearchRegion) -> None:
    path = Path(config.TOLOKA_CONFIG_PATH)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[config.SEND_REGION_2_KEY] = region.__dict__
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Сохранена область отправки 2: %s", region)


def select_send_region_2_interactively() -> SearchRegion | None:
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
    save_send_region_2(region)
    return region


# ---------- КЛАСС WATCHER ----------
class TolokaWatcher:
    def __init__(self, on_play_started, on_play_finished, log=LOG.info):
        self.on_play_started = on_play_started
        self.on_play_finished = on_play_finished
        self.log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_state = TolokaState.UNKNOWN
        self._play_seen = False
        self._pause_seen = False
        self._templates = None
        self._last_log_time = 0

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
            self._pause_seen = False
        self._thread = threading.Thread(target=self._loop, args=(region,), daemon=True)
        self._thread.start()
        self.log("Watcher запущен, отслеживание состояния плеера")
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self.log("Watcher остановлен")

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
        self.log("Шаблоны иконок загружены")

    def _loop(self, region: SearchRegion) -> None:
        import cv2
        import mss
        with mss.mss() as sct:
            while not self._stop.is_set():
                shot = sct.grab(region.as_mss())
                frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2GRAY)
                state, score, best_state = self._detect_with_score(frame, cv2)
                now = time.time()
                if state != self._last_state or (now - self._last_log_time > 5.0):
                    self._last_log_time = now
                    if state == TolokaState.UNKNOWN and best_state != TolokaState.UNKNOWN:
                        self.log(f"Обнаружено состояние: UNKNOWN (лучший шаблон {best_state.value} с оценкой {score:.2f}, порог {config.TOLOKA_TEMPLATE_THRESHOLD})")
                    else:
                        self.log(f"Обнаружено состояние: {state.value} (оценка {score:.2f})")
                if state != self._last_state:
                    self._last_state = state
                self._handle_state(state)
                self._stop.wait(config.TOLOKA_WATCH_INTERVAL_SECONDS)

    def _detect_with_score(self, frame, cv2):
        best_score = 0.0
        best_state = TolokaState.UNKNOWN
        for state, template in self._templates or []:
            if frame.shape[0] < template.shape[0] or frame.shape[1] < template.shape[1]:
                continue
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            max_val = float(result.max())
            if max_val > best_score:
                best_score = max_val
                best_state = state
        if best_score >= config.TOLOKA_TEMPLATE_THRESHOLD:
            return best_state, best_score, best_state
        return TolokaState.UNKNOWN, best_score, best_state

    def _detect(self, frame, cv2) -> TolokaState:
        state, _, _ = self._detect_with_score(frame, cv2)
        return state

    def _handle_state(self, state: TolokaState) -> None:
        if state == TolokaState.PAUSE:
            if self._play_seen:
                self._play_seen = False
                self._pause_seen = True
                self.log("Событие: начало воспроизведения (PLAY->PAUSE)")
                self.on_play_started()
            else:
                self._pause_seen = True
        elif state == TolokaState.PLAY:
            if self._pause_seen:
                self._pause_seen = False
                self._play_seen = True
                self.log("Событие: окончание воспроизведения (PAUSE->PLAY)")
                self.on_play_finished()
            else:
                self._play_seen = True
        elif state == TolokaState.LOADING:
            self._play_seen = False
            self._pause_seen = False
            self.log("Состояние LOADING – сброс флагов")

    def get_category_region(self, category: int) -> SearchRegion | None:
        return load_category_region(category)

    def select_category_region(self, category: int) -> bool:
        region = select_category_region_interactively(category)
        return region is not None