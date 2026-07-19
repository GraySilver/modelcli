"""Smoke tests for modelcli CLI — these verify wiring, not model quality.

Model downloads are avoided; these only test CLI help/arg parsing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLI = [sys.executable, "-m", "modelcli.cli"]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*CLI, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_top_level_help() -> None:
    r = _run("--help")
    assert r.returncode == 0
    assert "ocr" in r.stdout
    assert "asr" in r.stdout
    assert "tts" in r.stdout
    assert "models" in r.stdout


def test_ocr_help() -> None:
    r = _run("ocr", "--help")
    assert r.returncode == 0
    assert "image" in r.stdout.lower()


def test_asr_help() -> None:
    r = _run("asr", "--help")
    assert r.returncode == 0
    assert "audio" in r.stdout.lower()


def test_tts_help() -> None:
    r = _run("tts", "--help")
    assert r.returncode == 0
    assert "text" in r.stdout.lower()


def test_models_help() -> None:
    r = _run("models", "--help")
    assert r.returncode == 0


def test_missing_image_errors() -> None:
    r = _run("ocr", "/nonexistent/path.png")
    assert r.returncode != 0
