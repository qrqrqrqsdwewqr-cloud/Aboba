"""Optional, explicit keyboard/site automation helpers."""
from __future__ import annotations

import time
import re
import uuid
import logging
from dataclasses import dataclass

import config
from toloka_watcher import load_category_region, SearchRegion

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutomationResult:
    ok: bool
    message: str


def _import_pyautogui():
    try:
        import pyautogui
        return pyautogui, None
    except Exception as exc:
        return None, exc


def try_activate_window_by_title(title_substr: str | None) -> bool:
    if not title_substr:
        return False
    try:
        import pygetwindow as gw
    except Exception:
        return False
    try:
        wins = gw.getWindowsWithTitle(title_substr)
        if not wins:
            wins = [w for w in gw.getAllWindows() if title_substr.lower() in (w.title or "").lower()]
        if not wins:
            return False
        win = wins[0]
        try:
            win.activate()
        except Exception:
            win.minimize(); time.sleep(0.05); win.maximize()
        return True
    except Exception:
        return False


def category_key(category) -> str | None:
    if category in config.CATEGORY_KEYS:
        return config.CATEGORY_KEYS[category]
    if isinstance(category, str):
        lowered = category.strip().lower()
        return config.CATEGORY_KEYS.get(lowered)
    return None


def press_category(category, *, press_enter: bool = True, focus_title: str | None = None) -> AutomationResult:
    pyautogui, error = _import_pyautogui()
    if pyautogui is None:
        return AutomationResult(False, f"pyautogui unavailable: {error!r}")
    key = category_key(category)
    if not key:
        LOG.warning("Unknown category: %s", category)
        return AutomationResult(False, f"unknown category: {category}")
    try:
        try_activate_window_by_title(focus_title)
        LOG.info("Category: %s", category)
        LOG.info("Pressing key: %s", key)
        pyautogui.press(key)
        if press_enter:
            time.sleep(config.AUTOMATION_KEY_DELAY_SECONDS)
            pyautogui.press("enter")
        return AutomationResult(True, f"pressed category {category} with key {key}")
    except Exception as exc:
        return AutomationResult(False, f"keypress failed: {exc!r}")


# Функция send_text_to_site больше не используется в основном потоке,
# но оставлена для совместимости.
def send_text_to_site(text: str, *, category: int = 1, focus_title: str | None = None) -> AutomationResult:
    pyautogui, error = _import_pyautogui()
    if pyautogui is None:
        return AutomationResult(False, f"pyautogui unavailable: {error!r}")
    clean_text = sanitize_transcript_text(text)
    if not clean_text:
        return AutomationResult(False, "empty text")
    try:
        try_activate_window_by_title(focus_title)
        key = category_key(category)
        if not key:
            LOG.warning("Unknown category: %s", category)
            return AutomationResult(False, f"unknown category: {category}")
        LOG.info("Category: %s", category)
        LOG.info("Pressing key: %s", key)
        pyautogui.press(key)
        time.sleep(config.AUTOMATION_KEY_DELAY_SECONDS)
        pyautogui.press("enter")
        if not wait_for_text_field(pyautogui):
            return AutomationResult(False, "text field did not appear")
        if config.USE_CLIPBOARD_PASTE:
            import pyperclip
            pyperclip.copy(clean_text)
            pyautogui.hotkey("ctrl", "v")
        else:
            pyautogui.typewrite(clean_text, interval=0.01)
        if config.SITE_SUBMIT_AFTER_PASTE:
            time.sleep(config.SITE_AFTER_TYPE_DELAY_SECONDS)
            pyautogui.press("enter")
        return AutomationResult(True, "sent text to site")
    except Exception as exc:
        return AutomationResult(False, f"send failed: {exc!r}")


def sanitize_transcript_text(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"^\[[^\]]+\]\s*", "", value)
    value = re.sub(r"^\[Category:\s*\d+\]\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^[1-4]\s*(?=[^\d\s])", "", value)
    return value.strip()


def wait_for_text_field(pyautogui) -> bool:
    deadline = time.time() + config.TEXT_FIELD_READY_TIMEOUT_SECONDS
    while time.time() <= deadline:
        if _probe_text_field(pyautogui):
            return True
        time.sleep(config.TEXT_FIELD_READY_CHECK_INTERVAL_SECONDS)
    return False


def _probe_text_field(pyautogui) -> bool:
    if not config.USE_CLIPBOARD_PASTE:
        time.sleep(config.SITE_PRE_OPEN_DELAY_SECONDS)
        return True
    try:
        import pyperclip
        marker = f"__psunka_probe_{uuid.uuid4().hex}__"
        previous = pyperclip.paste()
        pyperclip.copy(marker)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.press("backspace")
        pyautogui.hotkey("ctrl", "v")
        time.sleep(config.AUTOMATION_KEY_DELAY_SECONDS)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.hotkey("ctrl", "c")
        focused_text = pyperclip.paste()
        pyautogui.press("backspace")
        pyperclip.copy(previous)
        return focused_text == marker
    except Exception:
        return False


# ---------- КЛИК ПО ОБЛАСТИ КАТЕГОРИИ ----------
def click_category_region(category: int) -> AutomationResult:
    pyautogui, error = _import_pyautogui()
    if pyautogui is None:
        return AutomationResult(False, f"pyautogui unavailable: {error!r}")

    region = load_category_region(category)
    if region is None:
        LOG.warning("Область для категории %d не выбрана", category)
        return AutomationResult(False, f"region for category {category} not set")

    try:
        x, y = region.center
        LOG.info("Клик по категории %d в координаты (%d, %d)", category, x, y)
        time.sleep(config.CATEGORY_CLICK_DELAY)
        pyautogui.click(x, y, button='left')
        return AutomationResult(True, f"clicked category {category} at ({x}, {y})")
    except Exception as exc:
        return AutomationResult(False, f"click failed: {exc!r}")


# ---------- ФУНКЦИИ ДЛЯ КНОПКИ "ОТПРАВИТЬ" ----------
def click_send_region() -> AutomationResult:
    """Кликнуть левой кнопкой мыши в центр области, сохранённой для кнопки 'Отправить'."""
    pyautogui, error = _import_pyautogui()
    if pyautogui is None:
        return AutomationResult(False, f"pyautogui unavailable: {error!r}")

    from toloka_watcher import load_send_region
    region = load_send_region()
    if region is None:
        LOG.warning("Область для кнопки 'Отправить' не выбрана")
        return AutomationResult(False, "send region not set")

    try:
        x, y = region.center
        LOG.info("Клик по области отправки в координаты (%d, %d)", x, y)
        time.sleep(config.CATEGORY_CLICK_DELAY)
        pyautogui.click(x, y, button='left')
        return AutomationResult(True, f"clicked send button at ({x}, {y})")
    except Exception as exc:
        return AutomationResult(False, f"click failed: {exc!r}")


def paste_text_from_clipboard(focus_title: str | None = None) -> AutomationResult:
    """Вставить текст из буфера обмена (Ctrl+V) с предварительной активацией окна и кликом для фокуса."""
    pyautogui, error = _import_pyautogui()
    if pyautogui is None:
        return AutomationResult(False, f"pyautogui unavailable: {error!r}")
    try:
        if focus_title:
            try_activate_window_by_title(focus_title)
            time.sleep(0.3)  # увеличенная задержка для активации

        # Клик левой кнопкой мыши в текущей позиции, чтобы переключить фокус
        pyautogui.click()
        time.sleep(0.1)

        # Нажатие Tab для переключения на поле ввода (если оно следующее)
        pyautogui.press("tab")
        time.sleep(0.1)

        # Используем hotkey для Ctrl+V
        pyautogui.hotkey("ctrl", "v")
        return AutomationResult(True, "pasted from clipboard")
    except Exception as exc:
        return AutomationResult(False, f"paste failed: {exc!r}")


# ---------- НОВЫЕ ФУНКЦИИ ДЛЯ КНОПКИ "ОТПРАВИТЬ 2" ----------
def click_send_region_2() -> AutomationResult:
    """Кликнуть левой кнопкой мыши в центр области, сохранённой для кнопки 'Отправить 2'."""
    pyautogui, error = _import_pyautogui()
    if pyautogui is None:
        return AutomationResult(False, f"pyautogui unavailable: {error!r}")

    from toloka_watcher import load_send_region_2
    region = load_send_region_2()
    if region is None:
        LOG.warning("Область для кнопки 'Отправить 2' не выбрана")
        return AutomationResult(False, "send region 2 not set")

    try:
        x, y = region.center
        LOG.info("Клик по области отправки 2 в координаты (%d, %d)", x, y)
        time.sleep(config.CATEGORY_CLICK_DELAY)
        pyautogui.click(x, y, button='left')
        return AutomationResult(True, f"clicked send button 2 at ({x}, {y})")
    except Exception as exc:
        return AutomationResult(False, f"click failed: {exc!r}")