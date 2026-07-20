from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "skills" / "modelcli" / "scripts" / "invoke.py"


def make_fake_modelcli(tmp_path: Path) -> Path:
    executable = tmp_path / "modelcli"
    executable.write_text(
        """#!/usr/bin/env python3
import os
import sys
import time

record = os.environ.get("FAKE_MODELCLI_RECORD")
if record:
    with open(record, "w", encoding="utf-8") as handle:
        handle.write("\\n".join(sys.argv[1:]))
delay = float(os.environ.get("FAKE_MODELCLI_DELAY", "0"))
if delay:
    time.sleep(delay)
stderr = os.environ.get("FAKE_MODELCLI_STDERR", "")
if stderr:
    sys.stderr.write(stderr)
sys.stdout.write(os.environ.get("FAKE_MODELCLI_STDOUT", "{}\\n"))
raise SystemExit(int(os.environ.get("FAKE_MODELCLI_EXIT", "0")))
"""
    )
    executable.chmod(0o755)
    return executable


def envelope(*, ok: bool = True) -> str:
    value = {
        "schema_version": "1",
        "ok": ok,
        "operation": "detect",
        "meta": {"modelcli_version": "test", "elapsed_ms": 1},
    }
    if ok:
        value["result"] = {"detections": []}
    else:
        value["error"] = {"code": "MODEL_ERROR", "message": "failed", "retryable": False}
    return json.dumps(value, separators=(",", ":")) + "\n"


def run_wrapper(tmp_path: Path, *arguments: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    executable = make_fake_modelcli(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "MODELCLI_BIN": str(executable),
            "FAKE_MODELCLI_STDOUT": envelope(),
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(WRAPPER), *arguments],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_wrapper_preserves_json_and_adds_agent_global_options(tmp_path: Path) -> None:
    record = tmp_path / "arguments.txt"
    stdout = envelope()
    result = run_wrapper(
        tmp_path,
        "--",
        "detect",
        "/tmp/photo.jpg",
        extra_env={"FAKE_MODELCLI_STDOUT": stdout, "FAKE_MODELCLI_RECORD": str(record)},
    )

    assert result.returncode == 0
    assert result.stdout == stdout
    assert record.read_text().splitlines() == ["--json", "--allow-download", "detect", "/tmp/photo.jpg"]


def test_wrapper_preserves_modelcli_error_and_exit_code(tmp_path: Path) -> None:
    stdout = envelope(ok=False)
    result = run_wrapper(
        tmp_path,
        "--",
        "detect",
        "missing.jpg",
        extra_env={"FAKE_MODELCLI_STDOUT": stdout, "FAKE_MODELCLI_EXIT": "4"},
    )

    assert result.returncode == 4
    assert result.stdout == stdout


def test_wrapper_rejects_invalid_protocol(tmp_path: Path) -> None:
    result = run_wrapper(tmp_path, "--", "capabilities", extra_env={"FAKE_MODELCLI_STDOUT": "not-json\n"})

    assert result.returncode == 5
    assert json.loads(result.stdout)["error"]["code"] == "PROTOCOL_ERROR"
    assert "ModelCLI stdout: not-json" in result.stderr


def test_wrapper_enforces_one_wall_clock_timeout(tmp_path: Path) -> None:
    result = run_wrapper(
        tmp_path,
        "--timeout",
        "0.05",
        "--",
        "capabilities",
        extra_env={"FAKE_MODELCLI_DELAY": "1"},
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 124
    assert payload["error"] == {
        "code": "TIMEOUT",
        "message": "ModelCLI exceeded 0.05 seconds",
        "retryable": True,
    }


@pytest.mark.parametrize(
    "command",
    [
        ("detect", "image.jpg", "--force"),
        ("--debug", "doctor", "--deep"),
        ("--debug", "models", "remove", "detect"),
        ("models", "clean"),
        ("models", "install", "detect", "--refresh"),
    ],
)
def test_wrapper_requires_confirmation_for_sensitive_commands(tmp_path: Path, command: tuple[str, ...]) -> None:
    result = run_wrapper(tmp_path, "--", *command)

    assert result.returncode == 2
    assert json.loads(result.stdout)["error"]["code"] == "CONFIRMATION_REQUIRED"


def test_wrapper_runs_sensitive_command_after_approval(tmp_path: Path) -> None:
    result = run_wrapper(tmp_path, "--approve-sensitive", "--", "models", "clean")

    assert result.returncode == 0
    assert json.loads(result.stdout)["ok"] is True


def test_invalid_configured_binary_does_not_trigger_install(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["MODELCLI_BIN"] = str(tmp_path / "missing-modelcli")
    result = subprocess.run(
        [sys.executable, str(WRAPPER), "--", "capabilities"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 4
    assert json.loads(result.stdout)["error"]["code"] == "MODELCLI_NOT_FOUND"


def test_wrapper_installs_missing_modelcli(tmp_path: Path) -> None:
    fake = make_fake_modelcli(tmp_path)
    home = tmp_path / "home"
    installer = tmp_path / "install.sh"
    installer.write_text(
        """#!/bin/sh
set -eu
mkdir -p "$HOME/.local/bin"
cp "$FAKE_MODELCLI_SOURCE" "$HOME/.local/bin/modelcli"
chmod +x "$HOME/.local/bin/modelcli"
printf '%s\\n' 'installed fake modelcli'
"""
    )
    installer.chmod(0o755)
    env = os.environ.copy()
    env.pop("MODELCLI_BIN", None)
    env.update(
        {
            "HOME": str(home),
            "MODELCLI_INSTALL_SCRIPT": str(installer),
            "FAKE_MODELCLI_SOURCE": str(fake),
            "FAKE_MODELCLI_STDOUT": envelope(),
            "PATH": "/usr/bin:/bin",
        }
    )

    result = subprocess.run(
        [sys.executable, str(WRAPPER), "--", "capabilities"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["ok"] is True
    assert "installed fake modelcli" in result.stderr
    assert (home / ".local" / "bin" / "modelcli").is_file()


def test_installation_uses_same_wall_clock_timeout(tmp_path: Path) -> None:
    installer = tmp_path / "install.sh"
    installer.write_text("#!/bin/sh\nsleep 1\n")
    installer.chmod(0o755)
    env = os.environ.copy()
    env.pop("MODELCLI_BIN", None)
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "MODELCLI_INSTALL_SCRIPT": str(installer),
            "PATH": "/usr/bin:/bin",
        }
    )

    result = subprocess.run(
        [sys.executable, str(WRAPPER), "--timeout", "0.05", "--", "capabilities"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 124
    assert json.loads(result.stdout)["error"]["code"] == "TIMEOUT"


@pytest.mark.skipif(os.name == "nt", reason="SIGINT process behavior is POSIX-specific")
def test_wrapper_returns_json_when_interrupted(tmp_path: Path) -> None:
    executable = make_fake_modelcli(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "MODELCLI_BIN": str(executable),
            "FAKE_MODELCLI_STDOUT": envelope(),
            "FAKE_MODELCLI_DELAY": "10",
        }
    )
    process = subprocess.Popen(
        [sys.executable, str(WRAPPER), "--", "capabilities"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    time.sleep(0.1)
    process.send_signal(signal.SIGINT)
    stdout, _ = process.communicate(timeout=5)

    assert process.returncode == 130
    assert json.loads(stdout)["error"] == {
        "code": "INTERRUPTED",
        "message": "Operation interrupted",
        "retryable": True,
    }
