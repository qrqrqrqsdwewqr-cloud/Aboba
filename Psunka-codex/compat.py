"""Compatibility checks and dependency bootstrap."""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass

import config

LOG_PATH = Path("compat.log")
MIN_PYTHON = (3, 10)

logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class CompatStatus:
    ok: bool
    messages: list[str]
    restart_required: bool = False


def check_python() -> tuple[bool, str]:
    current = sys.version_info[:3]
    ok = current >= MIN_PYTHON
    return ok, f"Python {current[0]}.{current[1]}.{current[2]} ({'OK' if ok else 'too old'})"


def ensure_dependencies(auto_install: bool = True) -> CompatStatus:
    messages: list[str] = []
    restart_required = False
    py_ok, py_msg = check_python()
    messages.append(py_msg)
    if not py_ok:
        logging.error(py_msg)
        return CompatStatus(False, messages)

    missing = [pip_name for import_name, pip_name in config.REQUIRED_PACKAGES.items() if importlib.util.find_spec(import_name) is None]
    if not missing:
        messages.append("All dependencies are installed")
        return CompatStatus(True, messages)

    messages.append("Missing: " + ", ".join(missing))
    logging.warning(messages[-1])
    if not auto_install:
        return CompatStatus(False, messages)

    cmd = [sys.executable, "-m", "pip", "install", *missing]
    logging.info("Installing dependencies: %s", cmd)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    logging.info("pip stdout: %s", proc.stdout)
    logging.info("pip stderr: %s", proc.stderr)
    if proc.returncode != 0:
        messages.append("Dependency installation failed; see compat.log")
        return CompatStatus(False, messages)

    still_missing = [pip_name for import_name, pip_name in config.REQUIRED_PACKAGES.items() if importlib.util.find_spec(import_name) is None]
    if still_missing:
        messages.append("Still missing after install: " + ", ".join(still_missing))
        return CompatStatus(False, messages)

    restart_required = True
    messages.append("Dependencies installed successfully; restart recommended")
    return CompatStatus(True, messages, restart_required)


def restart_program() -> None:
    logging.info("Restarting program")
    subprocess.Popen([sys.executable, *sys.argv])
    sys.exit(0)
