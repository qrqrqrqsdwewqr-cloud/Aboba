"""PyQt GUI with theme and compatibility status."""
from __future__ import annotations

import asyncio
import queue
import sys
import threading
import time
import logging
from enum import Enum

from PyQt5 import QtCore, QtWidgets, QtGui

import compat
import config
from automation import press_category, click_category_region, click_send_region, click_send_region_2, paste_text_from_clipboard
from audio_client import client_loop, list_devices
from audio_utils import AudioFragmentStore
from classifier import ClassificationResult
from toloka_watcher import (
    TolokaWatcher,
    select_region_interactively,
    select_category_region_interactively,
    select_send_region_interactively,
    select_send_region_2_interactively
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LIGHT_STYLE = """QWidget { background: #fafafa; color: #111; } QTextEdit { background: white; }"""
DARK_STYLE = """QWidget { background: #202124; color: #f1f3f4; } QTextEdit, QLineEdit, QComboBox { background: #303134; color: #f1f3f4; } QPushButton { background: #3c4043; color: #f1f3f4; padding: 4px; }"""


# ========== ОТДЕЛЬНОЕ ОКНО ЛОГОВ ==========
class QtLogBridge(QtCore.QObject):
    log_signal = QtCore.pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, bridge: QtLogBridge):
        super().__init__()
        self.bridge = bridge
        self.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.bridge.log_signal.emit(msg)
        except Exception:
            self.handleError(record)


class LogWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Мониторинг работы приложения")
        self.resize(700, 400)
        layout = QtWidgets.QVBoxLayout(self)

        self.text_browser = QtWidgets.QTextEdit(self)
        self.text_browser.setReadOnly(True)
        self.text_browser.setFont(QtGui.QFont("Consolas", 9))
        layout.addWidget(self.text_browser)

        btn_layout = QtWidgets.QHBoxLayout()
        clear_btn = QtWidgets.QPushButton("Очистить лог")
        clear_btn.clicked.connect(self.text_browser.clear)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def append_text(self, text: str):
        self.text_browser.append(text)
        cursor = self.text_browser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.text_browser.setTextCursor(cursor)


# ========== ОСНОВНОЙ КЛАСС ==========
class AutomationState(str, Enum):
    IDLE = "IDLE"
    WAITING_FOR_PLAY = "WAITING_FOR_PLAY"
    RECORDING = "RECORDING"
    SENDING = "SENDING"
    PROCESSING = "PROCESSING"
    CLASSIFYING = "CLASSIFYING"
    DONE = "DONE"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Классификатор речи (WebSocket STT)")
        self.resize(1000, 700)
        self.setMinimumWidth(760)

        # --- Логи в отдельное окно ---
        self.log_bridge = QtLogBridge()
        self.log_window = LogWindow(self)
        self.log_bridge.log_signal.connect(self.log_window.append_text)

        qt_handler = QtLogHandler(self.log_bridge)
        root_logger = logging.getLogger()
        root_logger.addHandler(qt_handler)

        # --- Остальные атрибуты ---
        self.gui_queue: queue.Queue = queue.Queue(maxsize=500)
        self.audio_store = AudioFragmentStore()
        self.client_thread: threading.Thread | None = None
        self.client_running = False
        self.last_text = ""
        self.last_category: int | None = None
        self.processed_count = 0

        self.start_event = threading.Event()
        self.finish_event = threading.Event()
        self.state = AutomationState.IDLE
        self.recording_start_time = 0

        self.watcher = TolokaWatcher(
            on_play_started=self.on_play_started,
            on_play_finished=self.on_play_finished,
            log=self.append_log
        )

        self._build_ui()
        self._connect()
        self._start_timer()
        self.run_compat_check()

        self.recording_timeout_timer = QtCore.QTimer()
        self.recording_timeout_timer.setSingleShot(True)
        self.recording_timeout_timer.timeout.connect(self._force_finish_recording)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Верхняя панель
        controls = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Старт")
        self.stop_btn = QtWidgets.QPushButton("Стоп")
        self.clear_btn = QtWidgets.QPushButton("Очистить")
        self.select_region_btn = QtWidgets.QPushButton("Выбрать область плеера")
        self.log_btn = QtWidgets.QPushButton("📊 Показать лог")
        self.reset_btn = QtWidgets.QPushButton("🔄 Сбросить статус")

        # Кнопки для выбора областей категорий
        self.select_cat1_btn = QtWidgets.QPushButton("📌 Кат.1")
        self.select_cat3_btn = QtWidgets.QPushButton("📌 Кат.3")
        self.select_cat4_btn = QtWidgets.QPushButton("📌 Кат.4")

        # Кнопки для выбора областей отправки
        self.select_send_btn = QtWidgets.QPushButton("📎 Область Отправить")
        self.select_send_2_btn = QtWidgets.QPushButton("📎 Область Отправить 2")   # <-- новая

        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(["Тёмная", "Светлая", "Системная"])

        self.process_only_checkbox = QtWidgets.QCheckBox("Обрабатывать только голос")
        self.process_only_checkbox.setChecked(config.PROCESS_ONLY_ON_VOICE)

        self.manual_category_combo = QtWidgets.QComboBox()
        self.manual_category_combo.addItems(["Авто", "1 - Русская", "2 - Иностранная", "3 - Неразборчивая", "4 - Шум"])

        self.keypress_checkbox = QtWidgets.QCheckBox("Нажимать категорию")
        self.keypress_checkbox.setChecked(config.ENABLE_KEYPRESS_ACTIONS)

        self.auto_send_checkbox = QtWidgets.QCheckBox("Автовставка (старое)")  # больше не используется
        self.auto_send_checkbox.setChecked(False)

        self.bandpass_checkbox = QtWidgets.QCheckBox("Полосовой фильтр")
        self.bandpass_checkbox.setChecked(config.ENABLE_BANDPASS_FILTER)

        self.notch_checkbox = QtWidgets.QCheckBox("Режекторный фильтр")
        self.notch_checkbox.setChecked(config.ENABLE_NOTCH_FILTER)

        self.send_btn = QtWidgets.QPushButton("Отправить последний")

        # Добавляем все виджеты в панель
        for widget in (
            self.start_btn,
            self.stop_btn,
            self.clear_btn,
            self.select_region_btn,
            self.select_cat1_btn,
            self.select_cat3_btn,
            self.select_cat4_btn,
            self.select_send_btn,
            self.select_send_2_btn,           # <-- добавлена
            self.log_btn,
            self.reset_btn,
            self.process_only_checkbox,
            QtWidgets.QLabel("Категория:"),
            self.manual_category_combo,
            QtWidgets.QLabel("Тема:"),
            self.theme_combo,
            self.keypress_checkbox,
            self.bandpass_checkbox,
            self.notch_checkbox,
            self.send_btn,
            self.auto_send_checkbox,
        ):
            controls.addWidget(widget)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.text_area = QtWidgets.QTextEdit(readOnly=True)
        layout.addWidget(self.text_area)

        # Нижняя панель со статусом и счётчиком
        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Статус: остановлен")
        self.count_label = QtWidgets.QLabel("Обработано: 0")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.count_label)
        layout.addLayout(status_layout)

        self.compat_label = QtWidgets.QLabel("Совместимость: проверка...")
        layout.addWidget(self.compat_label)

        replacement_row = QtWidgets.QHBoxLayout()
        self.replacement_edit = QtWidgets.QLineEdit()
        self.replacement_edit.setPlaceholderText("Добавить замену: что->на что")
        self.replacement_add_btn = QtWidgets.QPushButton("Добавить")
        replacement_row.addWidget(self.replacement_edit)
        replacement_row.addWidget(self.replacement_add_btn)
        layout.addLayout(replacement_row)

    def _connect(self) -> None:
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.clear_btn.clicked.connect(self.text_area.clear)
        self.log_btn.clicked.connect(self.show_log_window)
        self.reset_btn.clicked.connect(self._force_finish_recording)

        # Подключаем кнопки областей
        self.select_cat1_btn.clicked.connect(lambda: self.on_select_category_region(1))
        self.select_cat3_btn.clicked.connect(lambda: self.on_select_category_region(3))
        self.select_cat4_btn.clicked.connect(lambda: self.on_select_category_region(4))
        self.select_send_btn.clicked.connect(self.on_select_send_region)
        self.select_send_2_btn.clicked.connect(self.on_select_send_region_2)   # <-- новый коннект

        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        self.keypress_checkbox.stateChanged.connect(self.on_keypress_changed)
        self.auto_send_checkbox.stateChanged.connect(self.on_auto_send_changed)
        self.bandpass_checkbox.stateChanged.connect(self.on_bandpass_changed)
        self.notch_checkbox.stateChanged.connect(self.on_notch_changed)
        self.send_btn.clicked.connect(self.on_send_last)
        self.replacement_add_btn.clicked.connect(self.on_add_replacement)
        self.select_region_btn.clicked.connect(self.on_select_region)

    def _start_timer(self) -> None:
        self.timer = QtCore.QTimer()
        self.timer.setInterval(config.GUI_POLL_INTERVAL_MS)
        self.timer.timeout.connect(self.poll_queue)
        self.timer.start()

    def run_compat_check(self) -> None:
        status = compat.ensure_dependencies(auto_install=True)
        self.compat_label.setText("Совместимость: " + "; ".join(status.messages))
        self.apply_theme("Тёмная")

    def apply_theme(self, name: str) -> None:
        if name == "Светлая":
            self.setStyleSheet(LIGHT_STYLE)
        elif name == "Тёмная":
            self.setStyleSheet(DARK_STYLE)
        else:
            self.setStyleSheet("")

    def set_state(self, state: AutomationState) -> None:
        old_state = self.state
        self.state = state
        self.status_label.setText(f"Статус: {state.value}")
        self.append_log(f"Состояние: {old_state} -> {state.value}")

        if state == AutomationState.RECORDING:
            self.recording_start_time = time.time()
            self.recording_timeout_timer.start(60000)
        elif state in (AutomationState.SENDING, AutomationState.DONE, AutomationState.IDLE):
            self.recording_timeout_timer.stop()

    def _force_finish_recording(self):
        if self.state == AutomationState.RECORDING:
            self.append_log("⏱️ Принудительное завершение записи (таймаут или кнопка)")
            self.finish_event.set()
            self.set_state(AutomationState.SENDING)
        else:
            self.append_log("Запись не активна, сброс не требуется")

    def show_log_window(self) -> None:
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def on_start(self) -> None:
        if self.client_running:
            return
        self.client_running = True
        config.PROCESS_ONLY_ON_VOICE = self.process_only_checkbox.isChecked()
        self.set_state(AutomationState.WAITING_FOR_PLAY)
        self.client_thread = threading.Thread(
            target=lambda: asyncio.run(client_loop(
                config.DEVICE,
                self.gui_queue,
                self.audio_store,
                self.start_event,
                self.finish_event,
                self.append_log
            )),
            daemon=True
        )
        self.client_thread.start()
        self.watcher.start(sync=True)
        self.append_log("Клиентский поток запущен, ожидание начала воспроизведения (Play->Pause)")

    def on_stop(self) -> None:
        self.client_running = False
        self.set_state(AutomationState.IDLE)
        self.watcher.stop()
        self.append_log("Запрошена остановка")

    def poll_queue(self) -> None:
        try:
            while True:
                text, result = self.gui_queue.get_nowait()
                self.set_state(AutomationState.PROCESSING)
                selected = self.manual_category_combo.currentIndex()
                if selected:
                    result = type(result)(selected, f"manual category {selected}", result.rms, result.snr_db, result.spectral_entropy)
                    if config.ENABLE_KEYPRESS_ACTIONS or selected in (3, 4):
                        press_enter = True if selected in (3, 4) else config.AUTOMATION_PRESS_ENTER
                        press_category(selected, press_enter=press_enter, focus_title=config.SITE_WINDOW_TITLE)
                self.display_transcription(text, result)
                self.set_state(AutomationState.DONE)
        except queue.Empty:
            pass

    def display_transcription(self, text: str, result: ClassificationResult) -> None:
        ts = time.strftime("%H:%M:%S")
        self.text_area.append(f"[{ts}] [Категория: {result.category}] {text}")
        self.last_text = text
        self.last_category = result.category

        # Увеличиваем счётчик и обновляем метку
        self.processed_count += 1
        self.count_label.setText(f"Обработано: {self.processed_count}")

        # ---------- КЛИК ПО ОБЛАСТИ КАТЕГОРИИ ----------
        if config.ENABLE_CATEGORY_CLICK and result.category in config.CATEGORIES_TO_CLICK:
            self.append_log(f"Клик по категории {result.category}")
            click_result = click_category_region(result.category)
            self.append_log(f"Результат клика: {click_result.message}")

        # ---------- ЛОГИКА ДЛЯ КНОПКИ "ОТПРАВИТЬ" ----------
        if result.category == 1:
            # Категория 1: клик по области отправки + вставка текста
            self.append_log("Категория 1: клик по области отправки и вставка текста")
            click_res = click_send_region()
            self.append_log(f"Клик по области отправки: {click_res.message}")
            if click_res.ok:
                try:
                    import pyperclip
                    clean_text = self.last_text
                    pyperclip.copy(clean_text)
                    time.sleep(config.PASTE_DELAY_SECONDS)
                    paste_res = paste_text_from_clipboard(focus_title=config.SITE_WINDOW_TITLE)
                    self.append_log(f"Вставка: {paste_res.message}")

                    # ----- НОВАЯ ЛОГИКА: проверка количества слов и клик по второй области -----
                    word_count = len(self.last_text.split())
                    self.append_log(f"Количество слов: {word_count}")
                    if word_count > 3:
                        self.append_log("Слов > 3, клик по области отправки 2")
                        click_res_2 = click_send_region_2()
                        self.append_log(f"Клик по области отправки 2: {click_res_2.message}")
                except Exception as e:
                    self.append_log(f"Ошибка вставки: {e}")
        elif result.category == 4:
            # Категория 4: только клик по области отправки (без вставки)
            self.append_log("Категория 4: клик по области отправки (без вставки)")
            click_res = click_send_region()
            self.append_log(f"Клик по области отправки: {click_res.message}")
        # Для категорий 2 и 3 ничего не делаем

    def append_log(self, msg: str) -> None:
        logging.info(msg)

    def on_keypress_changed(self, state: int) -> None:
        config.ENABLE_KEYPRESS_ACTIONS = bool(state)
        self.append_log(f"Автонажатие категории: {config.ENABLE_KEYPRESS_ACTIONS}")

    def on_auto_send_changed(self, state: int) -> None:
        self.append_log("Флаг автовставки больше не используется (переключено на клик по области отправки)")

    def on_bandpass_changed(self, state: int) -> None:
        config.ENABLE_BANDPASS_FILTER = bool(state)
        self.append_log(f"Bandpass filter set to {config.ENABLE_BANDPASS_FILTER}")

    def on_notch_changed(self, state: int) -> None:
        config.ENABLE_NOTCH_FILTER = bool(state)
        self.append_log(f"Notch filter set to {config.ENABLE_NOTCH_FILTER}")

    def on_send_last(self) -> None:
        # Ручная отправка последнего текста – используем тот же механизм, что и авто
        if self.last_category != 1:
            self.append_log("Отправка пропущена: последняя категория не 1")
            return
        self.append_log("Ручная отправка (категория 1) – клик по области отправки + вставка")
        click_res = click_send_region()
        self.append_log(f"Клик по области отправки: {click_res.message}")
        if click_res.ok:
            try:
                import pyperclip
                pyperclip.copy(self.last_text)
                time.sleep(config.PASTE_DELAY_SECONDS)
                paste_res = paste_text_from_clipboard(focus_title=config.SITE_WINDOW_TITLE)
                self.append_log(f"Вставка: {paste_res.message}")

                # ----- НОВАЯ ЛОГИКА ДЛЯ РУЧНОЙ ОТПРАВКИ -----
                word_count = len(self.last_text.split())
                self.append_log(f"Количество слов: {word_count}")
                if word_count > 3:
                    self.append_log("Слов > 3, клик по области отправки 2")
                    click_res_2 = click_send_region_2()
                    self.append_log(f"Клик по области отправки 2: {click_res_2.message}")
            except Exception as e:
                self.append_log(f"Ошибка вставки: {e}")

    def on_add_replacement(self) -> None:
        text = self.replacement_edit.text().strip()
        if "->" not in text:
            self.append_log("Неверный формат замены. Используйте что->на что")
            return
        source, target = [part.strip() for part in text.split("->", 1)]
        if not source:
            self.append_log("Источник замены пуст")
            return
        config.REPLACEMENTS[source.lower()] = target
        self.replacement_edit.clear()
        self.append_log(f"Замена добавлена: '{source}' -> '{target}'")

    # ---------- МЕТОДЫ ВЫБОРА ОБЛАСТЕЙ ----------
    def on_select_category_region(self, category: int) -> None:
        """Выбрать область для клика по категории"""
        self.watcher.stop()
        self.append_log(f"Выбор области для категории {category}...")
        region = select_category_region_interactively(category)
        if region:
            self.append_log(f"Область для категории {category} выбрана: {region}")
        else:
            self.append_log(f"Область для категории {category} не выбрана")
        if self.client_running:
            self.watcher.start(sync=True)

    def on_select_region(self) -> None:
        """Выбор области плеера (оригинал)"""
        self.watcher.stop()
        region = select_region_interactively()
        self.append_log(f"Выбрана область: {region}" if region else "Область не выбрана")
        if self.client_running:
            self.watcher.start(sync=True)

    def on_select_send_region(self) -> None:
        """Выбрать область для кнопки 'Отправить'"""
        self.watcher.stop()
        self.append_log("Выбор области для кнопки 'Отправить'...")
        region = select_send_region_interactively()
        if region:
            self.append_log(f"Область отправки выбрана: {region}")
        else:
            self.append_log("Область отправки не выбрана")
        if self.client_running:
            self.watcher.start(sync=True)

    def on_select_send_region_2(self) -> None:
        """Выбрать область для кнопки 'Отправить 2'"""
        self.watcher.stop()
        self.append_log("Выбор области для кнопки 'Отправить 2'...")
        region = select_send_region_2_interactively()
        if region:
            self.append_log(f"Область отправки 2 выбрана: {region}")
        else:
            self.append_log("Область отправки 2 не выбрана")
        if self.client_running:
            self.watcher.start(sync=True)

    def on_play_started(self):
        self.set_state(AutomationState.RECORDING)
        self.append_log("Начало воспроизведения (Play->Pause) – начинаем запись")
        self.start_event.set()

    def on_play_finished(self):
        self.set_state(AutomationState.SENDING)
        self.append_log("Конец воспроизведения (Pause->Play) – отправляем аудио")
        self.finish_event.set()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if config.DEVICE is None:
        try:
            list_devices()
        except Exception:
            pass
    sys.exit(app.exec_())