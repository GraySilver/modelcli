"""Static capabilities and environment diagnostics."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

from modelcli import __version__
from modelcli.config import CACHE_ROOT
from modelcli.models.lifecycle import ModelTarget, get_model_status, list_models, verify_models
from modelcli.protocol import SCHEMA_VERSION


def capabilities() -> dict[str, Any]:
    return {
        "cli_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "global_options": ["--json", "--allow-download", "--debug", "--version"],
        "commands": {
            "ocr": {"options": ["--out", "--markdown", "--draw-boxes", "--force"]},
            "asr": {"options": ["--out", "--lang", "--no-vad", "--timestamps", "--emotion"]},
            "tts": {"options": ["--out", "--prompt-audio", "--max-duration", "--play", "--force"]},
            "models": {"actions": ["list", "install", "remove", "verify", "prefetch", "clean"]},
            "doctor": {"options": ["--deep"]},
        },
        "ocr": {
            "model": "PP-OCRv4 mobile",
            "languages": ["zh", "en"],
            "structured_lines": True,
            "annotated_image": True,
        },
        "asr": {
            "model": "SenseVoiceSmall INT8 ONNX",
            "languages": ["auto", "zh", "en", "yue", "ja", "ko"],
            "vad": True,
            "emotion_and_events": True,
        },
        "tts": {
            "model": "MOSS-TTS-Nano",
            "mode": "voice_clone",
            "sample_rate": 48000,
            "channels": 2,
            "prompt_audio": "optional; an installed default prompt is used when omitted",
        },
        "runtime": runtime_info(),
        "cache_directory": str(CACHE_ROOT.resolve()),
        "models": [model.to_dict() for model in list_models()],
        "agent_policy": {
            "implicit_download": False,
            "allow_download_option": "--allow-download",
            "inference_timeout_owner": "caller",
            "reserved_timeout_exit_code": 124,
        },
    }


def doctor(*, deep: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    dependencies = (
        "typer",
        "rapidocr_onnxruntime",
        "funasr_onnx",
        "onnxruntime",
        "torch",
        "transformers",
        "soundfile",
    )
    for dependency in dependencies:
        available = importlib.util.find_spec(dependency) is not None
        checks.append(_check(f"dependency:{dependency}", available, "available" if available else "missing"))

    checks.append(_writable_check("cache", CACHE_ROOT))
    checks.append(_writable_check("temporary", Path(tempfile.gettempdir())))
    checks.append(_writable_check("output", Path.cwd()))

    for model in list_models():
        if not model.managed:
            checks.append(_check(f"model:{model.name}", True, "bundled"))
            continue
        installed = model.status == "installed"
        checks.append(_check(f"model:{model.name}:files", installed, model.status))
        if not installed:
            checks.append(_check(f"model:{model.name}:manifest", False, "not verified"))
            continue
        try:
            verification = verify_models(ModelTarget(model.name))[0]
        except Exception as exc:
            checks.append(_check(f"model:{model.name}:manifest", False, str(exc)))
        else:
            checks.append(
                _check(
                    f"model:{model.name}:manifest",
                    True,
                    f"verified {verification['files_checked']} files",
                )
            )
        if deep:
            checks.append(_deep_model_check(model.name))

    return {
        "healthy": all(check["ok"] for check in checks),
        "deep": deep,
        "checks": checks,
        "runtime": runtime_info(),
    }


def runtime_info() -> dict[str, Any]:
    torch_info: dict[str, Any] = {"available": False, "cuda_available": False, "device": "cpu"}
    try:
        import torch

        cuda = torch.cuda.is_available()
        torch_info = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": cuda,
            "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        }
    except Exception:
        pass

    providers: list[str] = []
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
    except Exception:
        pass

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch_info,
        "onnx_providers": providers,
    }


def _deep_model_check(name: str) -> dict[str, Any]:
    try:
        from modelcli.models.locking import model_lock

        with model_lock(name, CACHE_ROOT):
            if name == "asr":
                from modelcli.models.lifecycle import prepare_asr_model

                prepare_asr_model(validate=True)
            else:
                from modelcli.models.lifecycle import prepare_tts_model

                prepare_tts_model(validate=True)
    except Exception as exc:
        return _check(f"model:{name}:deep_load", False, str(exc))
    return _check(f"model:{name}:deep_load", True, "loaded")


def _writable_check(name: str, directory: Path) -> dict[str, Any]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix=".modelcli-doctor-", dir=directory)
        os.close(fd)
        Path(path).unlink()
    except OSError as exc:
        return _check(f"writable:{name}", False, str(exc))
    return _check(f"writable:{name}", True, str(directory.resolve()))


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail}
