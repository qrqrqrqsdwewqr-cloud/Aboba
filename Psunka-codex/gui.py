"""PyQt GUI with GigaAM workflow and mouse-only site automation."""
from __future__ import annotations

import asyncio, queue, sys, threading, time, logging
from enum import Enum

from PyQt5 import QtCore, QtWidgets

import compat, config
from audio_client import client_loop, list_devices
from audio_utils import AudioFragmentStore
from classifier import ClassificationResult
from toloka_watcher import TolokaWatcher, select_region_interactively, load_region
from ui_controller import UIController

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LIGHT_STYLE = """QWidget { background: #fafafa; color: #111; } QTextEdit { background: white; }"""
DARK_STYLE = """QWidget { background: #202124; color: #f1f3f4; } QTextEdit, QLineEdit, QComboBox { background: #303134; color: #f1f3f4; } QPushButton { background: #3c4043; color: #f1f3f4; padding: 4px; }"""

class AutomationState(str, Enum):
    IDLE="IDLE"; WAITING_FOR_TASK="WAITING_FOR_TASK"; STARTING_PLAYBACK="STARTING_PLAYBACK"; RECORDING="RECORDING"; TRANSCRIBING="TRANSCRIBING"; DETECTING_LANGUAGE="DETECTING_LANGUAGE"; CLASSIFYING="CLASSIFYING"; SELECTING_CATEGORY="SELECTING_CATEGORY"; SUBMITTING_CATEGORY="SUBMITTING_CATEGORY"; WAITING_FOR_TEXT_FIELD="WAITING_FOR_TEXT_FIELD"; PASTING_TEXT="PASTING_TEXT"; SUBMITTING_TEXT="SUBMITTING_TEXT"; WAITING_FOR_NEXT_TASK="WAITING_FOR_NEXT_TASK"; PAUSED="PAUSED"; ERROR="ERROR"

ROI_SEQUENCE = [("player","1. Область Play/Pause"),("category_panel","2. Панель категорий"),("category_1","3. Чекбокс категории 1"),("category_2","4. Чекбокс категории 2"),("category_3","5. Чекбокс категории 3"),("category_4","6. Чекбокс категории 4"),("submit_button","7. Кнопка Отправить"),("text_field","8. Поле транскрипции")]

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Классификатор речи — GigaAM"); self.resize(1000, 700); self.setMinimumWidth(760)
        self.gui_queue: queue.Queue = queue.Queue(maxsize=500); self.audio_store = AudioFragmentStore(); self.client_thread: threading.Thread | None = None
        self.client_running = False; self.last_text = ""; self.last_category: int | None = None; self.state = AutomationState.IDLE
        self.recording_event = threading.Event(); self.finish_event = threading.Event(); self.stop_event = threading.Event(); self.ui_controller = UIController(log=self.append_log)
        self.watcher = TolokaWatcher(self.on_playback_started, self.on_playback_finished, log=logging.info, state_provider=lambda: self.state.value)
        self._build_ui(); self._connect(); self._start_timer(); self.run_compat_check()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self); controls = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Старт"); self.stop_btn = QtWidgets.QPushButton("Стоп"); self.clear_btn = QtWidgets.QPushButton("Очистить")
        self.select_region_btn = QtWidgets.QPushButton("Выбрать области"); self.check_regions_btn = QtWidgets.QPushButton("Проверить области")
        self.theme_combo = QtWidgets.QComboBox(); self.theme_combo.addItems(["Тёмная", "Светлая", "Системная"])
        self.mode_combo = QtWidgets.QComboBox(); self.mode_combo.addItems(["semi_auto", "automatic", "manual"]); self.mode_combo.setCurrentText(config.AUTOMATION_MODE)
        self.process_only_checkbox = QtWidgets.QCheckBox("Обрабатывать только голос"); self.process_only_checkbox.setChecked(config.PROCESS_ONLY_ON_VOICE)
        self.manual_category_combo = QtWidgets.QComboBox(); self.manual_category_combo.addItems(["Авто", "1 - Русская", "2 - Иностранная", "3 - Неразборчивая", "4 - Шум"])
        self.bandpass_checkbox = QtWidgets.QCheckBox("Полосовой фильтр"); self.bandpass_checkbox.setChecked(config.ENABLE_BANDPASS_FILTER)
        self.notch_checkbox = QtWidgets.QCheckBox("Режекторный фильтр"); self.notch_checkbox.setChecked(config.ENABLE_NOTCH_FILTER)
        for widget in (self.start_btn,self.stop_btn,self.clear_btn,self.select_region_btn,self.check_regions_btn,self.process_only_checkbox,QtWidgets.QLabel("Категория:"),self.manual_category_combo,QtWidgets.QLabel("Режим:"),self.mode_combo,QtWidgets.QLabel("Тема:"),self.theme_combo,self.bandpass_checkbox,self.notch_checkbox): controls.addWidget(widget)
        controls.addStretch(1); layout.addLayout(controls); self.text_area = QtWidgets.QTextEdit(readOnly=True); layout.addWidget(self.text_area)
        self.status_label = QtWidgets.QLabel("Статус: остановлен"); self.compat_label = QtWidgets.QLabel("Совместимость: проверка..."); layout.addWidget(self.status_label); layout.addWidget(self.compat_label)
        row = QtWidgets.QHBoxLayout(); self.replacement_edit = QtWidgets.QLineEdit(); self.replacement_edit.setPlaceholderText("Добавить замену: что->на что"); self.replacement_add_btn = QtWidgets.QPushButton("Добавить"); row.addWidget(self.replacement_edit); row.addWidget(self.replacement_add_btn); layout.addLayout(row)

    def _connect(self) -> None:
        self.start_btn.clicked.connect(self.on_start); self.stop_btn.clicked.connect(self.on_stop); self.clear_btn.clicked.connect(self.text_area.clear); self.theme_combo.currentTextChanged.connect(self.apply_theme)
        self.bandpass_checkbox.stateChanged.connect(self.on_bandpass_changed); self.notch_checkbox.stateChanged.connect(self.on_notch_changed); self.replacement_add_btn.clicked.connect(self.on_add_replacement)
        self.select_region_btn.clicked.connect(self.on_select_regions); self.check_regions_btn.clicked.connect(self.on_check_regions); self.mode_combo.currentTextChanged.connect(self.on_mode_changed)

    def _start_timer(self) -> None:
        self.timer = QtCore.QTimer(); self.timer.setInterval(config.GUI_POLL_INTERVAL_MS); self.timer.timeout.connect(self.poll_queue); self.timer.start()

    def run_compat_check(self) -> None:
        status = compat.ensure_dependencies(auto_install=True); self.compat_label.setText("Совместимость: " + "; ".join(status.messages)); self.apply_theme("Тёмная")
    def apply_theme(self, name: str) -> None: self.setStyleSheet(LIGHT_STYLE if name == "Светлая" else DARK_STYLE if name == "Тёмная" else "")
    def set_state(self, state: AutomationState) -> None: self.state = state; self.status_label.setText(f"Статус: {state.value}")

    def on_start(self) -> None:
        if self.client_running: return
        self.client_running = True; self.stop_event.clear(); config.PROCESS_ONLY_ON_VOICE = self.process_only_checkbox.isChecked(); config.AUTOMATION_MODE = self.mode_combo.currentText()
        self.set_state(AutomationState.WAITING_FOR_TASK)
        self.client_thread = threading.Thread(target=lambda: asyncio.run(client_loop(config.DEVICE, self.gui_queue, self.audio_store, self.recording_event, self.finish_event, self.stop_event, logging.info)), daemon=True); self.client_thread.start()
        self.watcher.start(sync=True); self.append_log("Клиентский поток GigaAM запущен")
        if config.AUTOMATION_MODE == "automatic": threading.Thread(target=self._start_playback_cycle, daemon=True).start()

    def on_stop(self) -> None:
        self.client_running = False; self.stop_event.set(); self.watcher.stop(); self.set_state(AutomationState.IDLE); self.append_log("Запрошена остановка")

    def poll_queue(self) -> None:
        try:
            while True:
                item = self.gui_queue.get_nowait(); text, result = item[0], item[1]
                raw_text = item[2] if len(item) > 2 else text; selected = self.manual_category_combo.currentIndex()
                if selected: result = type(result)(selected, f"manual category {selected}", result.rms, result.snr_db, result.spectral_entropy, (selected,))
                self.display_transcription(text, result)
                if config.ENABLE_MOUSE_AUTOMATION and config.AUTOMATION_MODE != "manual": threading.Thread(target=self._submit_result, args=(raw_text, result), daemon=True).start()
        except queue.Empty: pass

    def display_transcription(self, text: str, result: ClassificationResult) -> None:
        ts = time.strftime("%H:%M:%S"); cats = "+".join(map(str, result.selected_categories)); self.text_area.append(f"[{ts}] [Категория: {cats}] {text}"); self.last_text = text; self.last_category = result.category

    def _start_playback_cycle(self) -> None:
        if self.state in (AutomationState.PAUSED, AutomationState.ERROR): return
        self.set_state(AutomationState.STARTING_PLAYBACK)
        if not self.ui_controller.click_play(): self.set_state(AutomationState.ERROR); return

    def on_playback_started(self) -> None:
        if self.state in (AutomationState.PAUSED, AutomationState.ERROR): return
        self.set_state(AutomationState.RECORDING); self.recording_event.set(); self.append_log("Подтверждено начало воспроизведения")

    def on_playback_finished(self) -> None:
        if self.state != AutomationState.RECORDING: return
        self.set_state(AutomationState.TRANSCRIBING); self.finish_event.set(); self.append_log("Подтверждено окончание воспроизведения")

    def _submit_result(self, text: str, result: ClassificationResult) -> None:
        categories = list(result.selected_categories); self.set_state(AutomationState.SELECTING_CATEGORY)
        if not self.ui_controller.select_categories(categories): self.set_state(AutomationState.ERROR); return
        self.set_state(AutomationState.SUBMITTING_CATEGORY)
        if not self.ui_controller.click_submit(): self.set_state(AutomationState.ERROR); return
        if config.CATEGORY_RUSSIAN in categories:
            self.set_state(AutomationState.WAITING_FOR_TEXT_FIELD)
            if not self.ui_controller.wait_for_text_field(): self.set_state(AutomationState.ERROR); return
            self.set_state(AutomationState.PASTING_TEXT)
            if not self.ui_controller.paste_text(text): self.set_state(AutomationState.ERROR); return
            self.set_state(AutomationState.SUBMITTING_TEXT)
            if not self.ui_controller.click_submit(): self.set_state(AutomationState.ERROR); return
        self.set_state(AutomationState.WAITING_FOR_NEXT_TASK); self.ui_controller.wait_for_next_task(); self.set_state(AutomationState.WAITING_FOR_TASK)
        if config.AUTOMATION_MODE == "automatic": self._start_playback_cycle()

    def append_log(self, msg: str) -> None: logging.info(msg)
    def on_mode_changed(self, value: str) -> None: config.AUTOMATION_MODE = value
    def on_bandpass_changed(self, state: int) -> None: config.ENABLE_BANDPASS_FILTER = bool(state); self.append_log(f"Bandpass filter set to {config.ENABLE_BANDPASS_FILTER}")
    def on_notch_changed(self, state: int) -> None: config.ENABLE_NOTCH_FILTER = bool(state); self.append_log(f"Notch filter set to {config.ENABLE_NOTCH_FILTER}")

    def on_add_replacement(self) -> None:
        text = self.replacement_edit.text().strip()
        if "->" not in text: self.append_log("Неверный формат замены. Используйте что->на что"); return
        source, target = [part.strip() for part in text.split("->", 1)]
        if source: config.REPLACEMENTS[source.lower()] = target; self.replacement_edit.clear(); self.append_log(f"Замена добавлена: '{source}' -> '{target}'")

    def on_select_regions(self) -> None:
        self.watcher.stop()
        for name, label in ROI_SEQUENCE:
            QtWidgets.QMessageBox.information(self, "Настройка ROI", label)
            region = select_region_interactively(name); self.append_log(f"{label}: {region}" if region else f"{label}: не выбрана")
        if self.client_running: self.watcher.start(sync=True)

    def on_check_regions(self) -> None:
        lines = []
        for name, label in ROI_SEQUENCE: lines.append(f"{label}: {load_region(name)}")
        try:
            states = self.ui_controller.check_regions(); lines += [f"{k}: {v}" for k, v in states.items()]
        except Exception as exc: lines.append(f"Проверка шаблонов недоступна: {exc!r}")
        QtWidgets.QMessageBox.information(self, "Проверка областей", "\n".join(lines))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_F12:
            self.on_stop(); self.set_state(AutomationState.ERROR); return
        if event.key() == QtCore.Qt.Key_F11:
            self.set_state(AutomationState.PAUSED if self.state != AutomationState.PAUSED else AutomationState.WAITING_FOR_TASK); return
        super().keyPressEvent(event)

def main() -> None:
    app = QtWidgets.QApplication(sys.argv); window = MainWindow(); window.show()
    if config.DEVICE is None:
        try: list_devices()
        except Exception: pass
    sys.exit(app.exec_())
