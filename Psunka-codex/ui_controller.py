"""Single mouse-only control point for the Toloka web UI."""
from __future__ import annotations

import random, time
from dataclasses import dataclass
from pathlib import Path

import config
from clipboard_utils import copy_transcript
from toloka_watcher import SearchRegion, load_region

@dataclass(frozen=True)
class TemplateMatch:
    found: bool
    score: float
    center_x: int | None = None
    center_y: int | None = None
    scale: float | None = None

ELEMENTS = {"play":"play.png", "pause":"pause.png", "loading":"loading.png", "category_panel":"category_panel.png", "submit_button":"send_button.png", "text_field":"text_field.png"}
REGION_BY_ELEMENT = {"play":"player", "pause":"player", "loading":"player", "category_panel":"category_panel", "submit_button":"submit_button", "text_field":"text_field"}

class UIController:
    def __init__(self, log=print):
        self.log = log
        self.template_dir = Path(__file__).resolve().parent / "templates"
        self._cv2 = None; self._mss = None; self._pyautogui = None

    def _load_runtime(self):
        if self._cv2 is None:
            import cv2, mss, pyautogui
            self._cv2 = cv2; self._mss = mss; self._pyautogui = pyautogui
            self._pyautogui.FAILSAFE = bool(config.ENABLE_FAILSAFE)

    def find_element(self, name: str) -> TemplateMatch:
        self._load_runtime()
        filename = ELEMENTS.get(name, name if name.endswith(".png") else f"{name}.png")
        region_name = REGION_BY_ELEMENT.get(name, name)
        return self._match_template_in_region(filename, load_region(region_name))

    def wait_for_element(self, name: str, timeout: float) -> TemplateMatch:
        end = time.monotonic() + timeout; last = TemplateMatch(False, 0.0)
        while time.monotonic() < end:
            last = self.find_element(name)
            if last.found: return last
            time.sleep(config.UI_POLL_INTERVAL_SECONDS)
        return last

    def click_play(self) -> bool:
        for _ in range(config.CLICK_RETRY_COUNT):
            match = self.wait_for_element("play", config.ELEMENT_WAIT_TIMEOUT_SECONDS)
            if not match.found: continue
            self._click(match.center_x, match.center_y)
            if self.wait_for_element("pause", config.PLAY_START_TIMEOUT_SECONDS).found: return True
        self.log("Не удалось запустить Play: Pause не подтвердился")
        return False

    def get_category_state(self, category: int) -> bool | None:
        checked, unchecked = self._category_matches(category)
        if not checked.found and not unchecked.found: return None
        if abs(checked.score - unchecked.score) < 0.03: return None
        return checked.score > unchecked.score

    def set_category_state(self, category: int, checked: bool) -> bool:
        end = time.monotonic() + config.CHECKBOX_WAIT_TIMEOUT_SECONDS
        while time.monotonic() < end:
            state = self.get_category_state(category)
            if state is checked: return True
            if state is None:
                time.sleep(config.UI_POLL_INTERVAL_SECONDS); continue
            click_match = self._category_matches(category)[0 if state else 1]
            if click_match.found: self._click(click_match.center_x, click_match.center_y)
            elif config.ALLOW_CATEGORY_ROI_CENTER_FALLBACK:
                region = load_region(f"category_{category}")
                if region is None: return False
                self._click(*region.center)
            else: return False
            if self._wait_category_state(category, checked): return True
        return False

    def select_categories(self, categories: list[int]) -> bool:
        if tuple(sorted(categories)) not in config.SUPPORTED_CATEGORY_SETS: return False
        for attempt in range(2):
            for category in (1,2,3,4):
                if not self.set_category_state(category, category in set(categories)): return False
            if self.verify_exact_categories(categories): return True
            self.log(f"Категории не совпали, повтор настройки #{attempt + 1}")
        return False

    def verify_exact_categories(self, categories: list[int]) -> bool:
        target = set(categories)
        for category in (1,2,3,4):
            state = self.get_category_state(category)
            if state is None or state != (category in target): return False
        return True

    def click_submit(self) -> bool:
        match = self.wait_for_element("submit_button", config.ELEMENT_WAIT_TIMEOUT_SECONDS)
        if not match.found: return False
        self._click(match.center_x, match.center_y); return True

    def wait_for_text_field(self) -> bool:
        return self.wait_for_element("text_field", config.TEXT_FIELD_WAIT_TIMEOUT_SECONDS).found

    def paste_text(self, text: str) -> bool:
        match = self.wait_for_element("text_field", config.TEXT_FIELD_WAIT_TIMEOUT_SECONDS)
        if not match.found: return False
        self._click(match.center_x, match.center_y)
        self._load_runtime(); pg = self._pyautogui
        pg.hotkey("ctrl", "a"); time.sleep(config.MOUSE_AFTER_CLICK_DELAY_SECONDS)
        pg.press("backspace"); copy_transcript(text); pg.hotkey("ctrl", "v")
        return True

    def wait_for_next_task(self) -> bool:
        return self.wait_for_element("play", config.NEXT_TASK_WAIT_TIMEOUT_SECONDS).found

    def check_regions(self) -> dict[str, object]:
        result = {name: self.find_element(name) for name in ELEMENTS}
        for cat in (1,2,3,4): result[f"category_{cat}"] = self.get_category_state(cat)
        return result

    def _wait_category_state(self, category: int, checked: bool) -> bool:
        end = time.monotonic() + config.CHECKBOX_WAIT_TIMEOUT_SECONDS
        while time.monotonic() < end:
            if self.get_category_state(category) is checked: return True
            time.sleep(config.UI_POLL_INTERVAL_SECONDS)
        return False

    def _category_matches(self, category: int) -> tuple[TemplateMatch, TemplateMatch]:
        region = load_region(f"category_{category}")
        return (self._match_template_in_region(f"category_{category}_checked.png", region), self._match_template_in_region(f"category_{category}_unchecked.png", region))

    def _match_template_in_region(self, filename: str, region: SearchRegion | None) -> TemplateMatch:
        self._load_runtime(); cv2 = self._cv2
        if region is None: return TemplateMatch(False, 0.0)
        template = cv2.imread(str(self.template_dir / filename), cv2.IMREAD_GRAYSCALE)
        if template is None: return TemplateMatch(False, 0.0)
        with self._mss.mss() as sct:
            shot = sct.grab(region.as_mss())
        frame = cv2.cvtColor(__import__('numpy').array(shot), cv2.COLOR_BGRA2GRAY)
        best = TemplateMatch(False, 0.0)
        scales = config.TEMPLATE_SCALES if config.ENABLE_MULTISCALE_TEMPLATE_MATCHING else (1.0,)
        for scale in scales:
            w = max(1, int(template.shape[1] * scale)); h = max(1, int(template.shape[0] * scale))
            if w > frame.shape[1] or h > frame.shape[0] or w < 3 or h < 3: continue
            scaled = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            res = cv2.matchTemplate(frame, scaled, cv2.TM_CCOEFF_NORMED); _, score, _, loc = cv2.minMaxLoc(res)
            if score > best.score:
                best = TemplateMatch(score >= config.TOLOKA_TEMPLATE_THRESHOLD, float(score), region.left + loc[0] + w//2, region.top + loc[1] + h//2, scale)
        return best if best.found else TemplateMatch(False, best.score, best.center_x, best.center_y, best.scale)

    def _click(self, x: int | None, y: int | None) -> None:
        if x is None or y is None: return
        self._load_runtime(); pg = self._pyautogui
        dx = random.randint(-config.MOUSE_RANDOM_OFFSET_PX, config.MOUSE_RANDOM_OFFSET_PX) if config.MOUSE_RANDOM_OFFSET_PX else 0
        dy = random.randint(-config.MOUSE_RANDOM_OFFSET_PX, config.MOUSE_RANDOM_OFFSET_PX) if config.MOUSE_RANDOM_OFFSET_PX else 0
        if config.SMOOTH_MOUSE: pg.moveTo(x + dx, y + dy, duration=config.MOUSE_MOVE_DURATION_SECONDS)
        else: pg.moveTo(x + dx, y + dy)
        time.sleep(config.MOUSE_BEFORE_CLICK_DELAY_SECONDS); pg.click(button="left"); time.sleep(config.MOUSE_AFTER_CLICK_DELAY_SECONDS)
