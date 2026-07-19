"""Lifecycle management for downloadable modelcli models.

ASR = SenseVoice-Small (ModelScope).
TTS = MOSS-TTS-Nano (main model + MOSS-Audio-Tokenizer-Nano + default prompt).
OCR and VAD ship inside pip packages and are reported as ``bundled``.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from modelcli.config import (
    CACHE_ROOT,
    MOSS_DEFAULT_PROMPT_NAME,
    MOSS_DEFAULT_PROMPT_URL,
    MODELSCOPE_MOSS_AUDIO_TOKENIZER,
    MODELSCOPE_MOSS_TTS,
    MODELSCOPE_SENSEVOICE,
)


class ModelTarget(str, Enum):
    asr = "asr"
    tts = "tts"
    all = "all"


ModelStatusName = Literal["installed", "missing", "bundled"]


@dataclass(frozen=True)
class ModelStatus:
    name: str
    model: str
    status: ModelStatusName
    size_bytes: int
    managed: bool

    def to_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ModelActionResult:
    name: str
    status: ModelStatusName
    changed: bool
    size_bytes: int

    def to_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)


ASR_MODEL_NAME = "SenseVoiceSmall"
TTS_MODEL_NAME = "MOSS-TTS-Nano"

ASR_DOWNLOAD_FILES = (
    "config.yaml",
    "tokens.json",
    "model.pt",
    "am.mvn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
)
# funasr-onnx auto-exports model.onnx (+ optional external weights model.onnx.data).
ASR_REQUIRED_FILES = (
    ".download_ok",
    "config.yaml",
    "tokens.json",
    "model.onnx",
    "model.onnx.data",
)
# MOSS TTS sentinel files, all relative to CACHE_ROOT.
TTS_REQUIRED_FILES = (
    f"{MODELSCOPE_MOSS_TTS.replace('/', '__')}/.download_ok",
    f"{MODELSCOPE_MOSS_AUDIO_TOKENIZER.replace('/', '__')}/.download_ok",
    f"moss_prompts/{MOSS_DEFAULT_PROMPT_NAME}",
)


def asr_cache_dir() -> Path:
    return CACHE_ROOT / MODELSCOPE_SENSEVOICE.replace("/", "__")


def tts_cache_dir() -> Path:
    # TTS artifacts live under CACHE_ROOT across the main model dir, tokenizer
    # dir, and moss_prompts/; returns CACHE_ROOT so _path_size reports the
    # combined footprint for TTS-managed files.
    return CACHE_ROOT


def list_models() -> list[ModelStatus]:
    return [
        _asr_status(),
        _tts_status(),
        ModelStatus("ocr", "PP-OCRv4 mobile", "bundled", 0, False),
        ModelStatus("vad", "Silero VAD", "bundled", 0, False),
    ]


def get_model_status(name: str) -> ModelStatus:
    return next(model for model in list_models() if model.name == name)


def install_models(target: ModelTarget) -> list[ModelActionResult]:
    results: list[ModelActionResult] = []
    for name in _target_names(target):
        before = get_model_status(name)
        if name == "asr":
            prepare_asr_model(validate=True)
        else:
            prepare_tts_model()
        after = get_model_status(name)
        if after.status != "installed":
            raise RuntimeError(f"{name} model installation did not produce a complete cache")
        results.append(
            ModelActionResult(
                name=name,
                status=after.status,
                changed=before.status != "installed",
                size_bytes=after.size_bytes,
            )
        )
    return results


def remove_models(target: ModelTarget) -> list[ModelActionResult]:
    results: list[ModelActionResult] = []
    for name in _target_names(target):
        if name == "asr":
            changed = _rmtree_if_exists(asr_cache_dir())
            _remove_asr_locks()
            results.append(ModelActionResult(name, "missing", changed, 0))
        else:
            changed = _rmtree_if_exists(CACHE_ROOT / MODELSCOPE_MOSS_TTS.replace("/", "__"))
            changed |= _rmtree_if_exists(CACHE_ROOT / MODELSCOPE_MOSS_AUDIO_TOKENIZER.replace("/", "__"))
            prompt_dir = CACHE_ROOT / "moss_prompts"
            prompt = prompt_dir / MOSS_DEFAULT_PROMPT_NAME
            if prompt.exists():
                prompt.unlink()
                changed = True
            if prompt_dir.exists() and not any(prompt_dir.iterdir()):
                prompt_dir.rmdir()
            results.append(ModelActionResult(name, "missing", changed, 0))
    return results


def prepare_asr_model(*, validate: bool) -> Path:
    from funasr_onnx import SenseVoiceSmall

    from modelcli.models.cache import ensure_modelscope

    existing = asr_cache_dir()
    needs_repair = not all(
        _is_nonempty_file(existing / name) for name in ASR_DOWNLOAD_FILES
    )
    if needs_repair:
        (existing / ".download_ok").unlink(missing_ok=True)
    model_dir = ensure_modelscope(MODELSCOPE_SENSEVOICE)
    # Validate if explicitly requested, if we just repaired an incomplete
    # download, or if exported ONNX files are still missing.
    if (
        validate
        or needs_repair
        or not all(_is_nonempty_file(existing / name) for name in ASR_REQUIRED_FILES)
    ):
        SenseVoiceSmall(str(model_dir), batch_size=1)
    return model_dir


def prepare_tts_model() -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from modelcli.models.cache import ensure_file_from_url, ensure_modelscope

    ensure_file_from_url(
        MOSS_DEFAULT_PROMPT_URL,
        MOSS_DEFAULT_PROMPT_NAME,
        CACHE_ROOT / "moss_prompts",
    )
    main_dir = ensure_modelscope(MODELSCOPE_MOSS_TTS)
    ensure_modelscope(MODELSCOPE_MOSS_AUDIO_TOKENIZER)
    m = AutoModelForCausalLM.from_pretrained(str(main_dir), trust_remote_code=True)
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _target_names(target: ModelTarget) -> tuple[str, ...]:
    if target == ModelTarget.all:
        return (ModelTarget.asr.value, ModelTarget.tts.value)
    return (target.value,)


def _asr_status() -> ModelStatus:
    root = asr_cache_dir()
    installed = all(_is_nonempty_file(root / relative) for relative in ASR_REQUIRED_FILES)
    size = _path_size(root) if installed else 0
    for extra in ("model_quant.onnx",):
        f = root / extra
        if f.exists():
            size += f.stat().st_size
    return ModelStatus("asr", ASR_MODEL_NAME, "installed" if installed else "missing", size, True)


def _tts_status() -> ModelStatus:
    installed = all(
        _is_nonempty_file(CACHE_ROOT / relative) for relative in TTS_REQUIRED_FILES
    )
    size = 0
    if installed:
        size = sum(
            _path_size(CACHE_ROOT / relative.split("/")[0])
            for relative in TTS_REQUIRED_FILES
        )
    return ModelStatus("tts", TTS_MODEL_NAME, "installed" if installed else "missing", size, True)


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file() and not child.is_symlink()
    )


def _rmtree_if_exists(path: Path) -> bool:
    if path.exists():
        shutil.rmtree(path)
        return True
    return False


def _remove_asr_locks() -> None:
    lock_dir = CACHE_ROOT / ".lock"
    if not lock_dir.exists():
        return
    model_key = MODELSCOPE_SENSEVOICE.replace("/", "___")
    for lock_file in lock_dir.glob(f"model_{model_key}*"):
        if lock_file.is_file():
            lock_file.unlink()
