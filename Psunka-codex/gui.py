"""PyQt GUI with theme and compatibility status."""
from __future__ import annotations

import asyncio
import queue
import sys
import threading
import time
import logging

from PyQt5 import QtCore, QtWidgets

import compat
import config
from automation import press_category, send_text_to_site
from audio_client import client_loop, list_devices
from audio_utils import AudioFragmentStore
from classifier import ClassificationResult
from toloka_watcher import TolokaWatcher, select_region_interactively

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LIGHT_STYLE = """QWidget { background: #fafafa; color: #111; } QTextEdit { background: white; }"""
DARK_STYLE = """QWidget { background: #202124; color: #f1f3f4; } QTextEdit, QLineEdit, QComboBox { background: #303134; color: #f1f3f4; } QPushButton { background: #3c4043; color: #f1f3f4; padding: 4px; }"""


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Классификатор речи")
        self.resize(1000, 700)
        self.setMinimumWidth(760)
        self.gui_queue: queue.Queue = queue.Queue(maxsize=500)
        self.audio_store = AudioFragmentStore()
        self.client_thread: threading.Thread | None = None
        self.client_running = False
        self.last_text = ""
        self.last_category: int | None = None
        self.wait_user_confirmation = False
        self.process_event = threading.Event()
        self.watcher = TolokaWatcher(self.on_toloka_play_to_pause, log=logging.info)
        self._build_ui()
        self._connect()
        self._start_timer()
        self.run_compat_check()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Старт")
        self.stop_btn = QtWidgets.QPushButton("Стоп")
        self.clear_btn = QtWidgets.QPushButton("Очистить")
        self.select_region_btn = QtWidgets.QPushButton("Выбрать область")
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(["Тёмная", "Светлая", "Системная"])
        self.process_only_checkbox = QtWidgets.QCheckBox("Обрабатывать только голос")
        self.process_only_checkbox.setChecked(config.PROCESS_ONLY_ON_VOICE)
        self.manual_category_combo = QtWidgets.QComboBox()
        self.manual_category_combo.addItems(["Авто", "1 - Русская", "2 - Иностранная", "3 - Неразборчивая", "4 - Шум"])
        self.keypress_checkbox = QtWidgets.QCheckBox("Нажимать категорию")
        self.keypress_checkbox.setChecked(config.ENABLE_KEYPRESS_ACTIONS)
        self.auto_send_checkbox = QtWidgets.QCheckBox("Автовставка категории 1")
        self.auto_send_checkbox.setChecked(config.AUTO_SEND_RUSSIAN_TO_SITE)
        self.bandpass_checkbox = QtWidgets.QCheckBox("Полосовой фильтр")
        self.bandpass_checkbox.setChecked(config.ENABLE_BANDPASS_FILTER)
        self.notch_checkbox = QtWidgets.QCheckBox("Режекторный фильтр")
        self.notch_checkbox.setChecked(config.ENABLE_NOTCH_FILTER)
        self.send_btn = QtWidgets.QPushButton("Отправить последний")
        for widget in (
            self.start_btn,
            self.stop_btn,
            self.clear_btn,
            self.select_region_btn,
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
        self.status_label = QtWidgets.QLabel("Статус: остановлен")
        self.compat_label = QtWidgets.QLabel("Совместимость: проверка...")
        layout.addWidget(self.status_label)
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

    def on_start(self) -> None:
        if self.client_running:
            return
        self.client_running = True
        config.PROCESS_ONLY_ON_VOICE = self.process_only_checkbox.isChecked()
        self.status_label.setText("Статус: запущен")
        self.client_thread = threading.Thread(target=lambda: asyncio.run(client_loop(config.DEVICE, self.gui_queue, self.audio_store, self.process_event, logging.info)), daemon=True)
        self.client_thread.start()
        self.watcher.start(sync=True)
        self.append_log("Клиентский поток запущен")

    def on_stop(self) -> None:
        self.client_running = False
        self.status_label.setText("Статус: остановлен (перезапустите приложение для закрытия websocket)")
        self.watcher.stop()
        self.append_log("Запрошена остановка")

    def poll_queue(self) -> None:
        try:
            while True:
                text, result = self.gui_queue.get_nowait()
                selected = self.manual_category_combo.currentIndex()
                if selected:
                    result = type(result)(selected, f"manual category {selected}", result.rms, result.snr_db, result.spectral_entropy)
                    if config.ENABLE_KEYPRESS_ACTIONS or selected in (3, 4):
                        press_enter = True if selected in (3, 4) else config.AUTOMATION_PRESS_ENTER
                        press_category(selected, press_enter=press_enter, focus_title=config.SITE_WINDOW_TITLE)
                self.display_transcription(text, result)
        except queue.Empty:
            pass

    def display_transcription(self, text: str, result: ClassificationResult) -> None:
        ts = time.strftime("%H:%M:%S")
        self.text_area.append(f"[{ts}] [Категория: {result.category}] {text}")
        self.last_text = text
        self.last_category = result.category
        if result.category == 1 and config.AUTO_SEND_RUSSIAN_TO_SITE:
            self.watcher.stop()
            self.append_log("Переход в WAIT_USER_CONFIRMATION")
            send_text_to_site(text, category=1, focus_title=config.SITE_WINDOW_TITLE)
            self.wait_user_confirmation = True
            self.status_label.setText("Статус: ожидание подтверждения Enter")

    def append_log(self, msg: str) -> None:
        logging.info(msg)

    def on_keypress_changed(self, state: int) -> None:
        config.ENABLE_KEYPRESS_ACTIONS = bool(state)
        self.append_log(f"Автонажатие категории: {config.ENABLE_KEYPRESS_ACTIONS}")

    def on_auto_send_changed(self, state: int) -> None:
        config.AUTO_SEND_RUSSIAN_TO_SITE = bool(state)
        self.append_log(f"Автовставка категории 1: {config.AUTO_SEND_RUSSIAN_TO_SITE}")

    def on_bandpass_changed(self, state: int) -> None:
        config.ENABLE_BANDPASS_FILTER = bool(state)
        self.append_log(f"Bandpass filter set to {config.ENABLE_BANDPASS_FILTER}")

    def on_notch_changed(self, state: int) -> None:
        config.ENABLE_NOTCH_FILTER = bool(state)
        self.append_log(f"Notch filter set to {config.ENABLE_NOTCH_FILTER}")

    def on_send_last(self) -> None:
        if self.last_category != 1:
            self.append_log("Отправка пропущена: последняя категория не 1")
            return
        result = send_text_to_site(self.last_text, category=1, focus_title=config.SITE_WINDOW_TITLE)
        self.append_log(result.message)

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

    def on_toloka_play_to_pause(self) -> None:
        if self.wait_user_confirmation:
            return
        self.append_log("Разрешён запуск обработки после PLAY -> PAUSE")
        self.process_event.set()

    def on_select_region(self) -> None:
        self.watcher.stop()
        region = select_region_interactively()
        self.append_log(f"Выбрана область: {region}" if region else "Область не выбрана")
        if self.client_running and not self.wait_user_confirmation:
            self.watcher.start(sync=True)

    def keyPressEvent(self, event):
        if self.wait_user_confirmation and event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.append_log("Подтверждение пользователем получено")
            self.wait_user_confirmation = False
            self.process_event.clear()
            self.status_label.setText("Статус: запущен")
            self.watcher.start(sync=True)
            return
        super().keyPressEvent(event)


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
