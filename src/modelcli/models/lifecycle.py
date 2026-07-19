"""Lifecycle and integrity management for downloadable models."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from modelcli.config import (
    CACHE_ROOT,
    MOSS_DEFAULT_PROMPT_NAME,
    MOSS_DEFAULT_PROMPT_SHA256,
    MOSS_DEFAULT_PROMPT_URL,
    MODELSCOPE_MOSS_AUDIO_TOKENIZER,
    MODELSCOPE_MOSS_TTS,
    MODELSCOPE_REVISION,
    MODELSCOPE_SENSEVOICE,
    MODELSCOPE_SENSEVOICE_REVISION,
    SENSEVOICE_TOKENIZER_NAME,
    SENSEVOICE_TOKENIZER_SHA256,
    SENSEVOICE_TOKENIZER_URL,
)
from modelcli.errors import model_error
from modelcli.models.locking import model_lock
from modelcli.models.manifest import (
    create_manifest,
    load_manifest,
    manifest_path,
    sha256_file,
    verify_manifest,
)
from modelcli.protocol import current_runtime


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
    manifest_status: str = "not_applicable"
    verification_status: str = "not_applicable"
    requested_revision: str | None = None
    source_revision: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelActionResult:
    name: str
    status: ModelStatusName
    changed: bool
    size_bytes: int
    manifest_status: str = "missing"
    verification_status: str = "unverified"
    requested_revision: str | None = None
    source_revision: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


ASR_MODEL_NAME = "SenseVoiceSmall INT8 ONNX"
TTS_MODEL_NAME = "MOSS-TTS-Nano"

ASR_DOWNLOAD_FILES = (
    "config.yaml",
    "tokens.json",
    "model_quant.onnx",
    "am.mvn",
    "configuration.json",
)
ASR_REQUIRED_FILES = (
    ".download_ok",
    *ASR_DOWNLOAD_FILES,
    SENSEVOICE_TOKENIZER_NAME,
)
TTS_REQUIRED_FILES = (
    f"{MODELSCOPE_MOSS_TTS.replace('/', '__')}/.download_ok",
    f"{MODELSCOPE_MOSS_AUDIO_TOKENIZER.replace('/', '__')}/.download_ok",
    f"moss_prompts/{MOSS_DEFAULT_PROMPT_NAME}",
)


def asr_cache_dir(cache_root: Path | None = None) -> Path:
    return (cache_root or CACHE_ROOT) / MODELSCOPE_SENSEVOICE.replace("/", "__")


def tts_cache_dir(cache_root: Path | None = None) -> Path:
    return cache_root or CACHE_ROOT


def list_models() -> list[ModelStatus]:
    return [
        _managed_status("asr"),
        _managed_status("tts"),
        ModelStatus("ocr", "PP-OCRv4 mobile", "bundled", 0, False),
        ModelStatus("vad", "Silero VAD", "bundled", 0, False),
    ]


def get_model_status(name: str) -> ModelStatus:
    return next(model for model in list_models() if model.name == name)


def install_models(
    target: ModelTarget,
    *,
    refresh: bool = False,
) -> list[ModelActionResult]:
    results: list[ModelActionResult] = []
    for name in _target_names(target):
        with model_lock(name, CACHE_ROOT):
            before = _managed_status(name)
            if refresh:
                _refresh_model(name)
            elif name == "asr":
                prepare_asr_model(validate=True, allow_download=True)
            else:
                prepare_tts_model(validate=True, allow_download=True)
            verification = verify_manifest(CACHE_ROOT, name)
            after = _managed_status(name)
            if after.status != "installed":
                raise model_error(
                    "MODEL_INSTALL_FAILED",
                    f"Model '{name}' installation did not produce a complete cache",
                )
            results.append(
                ModelActionResult(
                    name=name,
                    status=after.status,
                    changed=refresh or before.status != "installed",
                    size_bytes=after.size_bytes,
                    manifest_status="present",
                    verification_status="verified",
                    requested_revision=verification["requested_revision"],
                    source_revision=verification["source_revision"],
                )
            )
    return results


def remove_models(target: ModelTarget) -> list[ModelActionResult]:
    results: list[ModelActionResult] = []
    for name in _target_names(target):
        with model_lock(name, CACHE_ROOT):
            changed = _remove_one(name)
            results.append(ModelActionResult(name, "missing", changed, 0))
    return results


def verify_models(target: ModelTarget) -> list[dict]:
    results: list[dict] = []
    for name in _target_names(target):
        with model_lock(name, CACHE_ROOT):
            if not _is_complete(name, CACHE_ROOT):
                raise model_error("MODEL_NOT_INSTALLED", f"Model '{name}' is not installed")
            manifest = load_manifest(CACHE_ROOT, name)
            if name == "asr" and (
                manifest is None or not _manifest_matches_standard("asr", manifest)
            ):
                _load_asr(asr_cache_dir(CACHE_ROOT))
            _adopt_manifest_if_needed(name, CACHE_ROOT)
            results.append(verify_manifest(CACHE_ROOT, name))
    return results


def prepare_asr_model(
    *,
    validate: bool,
    cache_root: Path | None = None,
    allow_download: bool = False,
) -> Path:
    root = cache_root or CACHE_ROOT
    if _is_complete("asr", root):
        manifest = load_manifest(root, "asr")
        loaded = False
        if manifest is None or not _manifest_matches_standard("asr", manifest):
            _load_asr(asr_cache_dir(root))
            loaded = True
            _adopt_manifest_if_needed("asr", root)
        verify_manifest(root, "asr")
        if validate and not loaded:
            _load_asr(asr_cache_dir(root))
        return asr_cache_dir(root)

    manifest = load_manifest(root, "asr")
    if manifest is not None and _manifest_matches_standard("asr", manifest):
        verify_manifest(root, "asr")

    _require_download_allowed("asr", cache_root=cache_root, explicit=allow_download)
    model_dir = _download_asr(root)
    _load_asr(model_dir)
    if not _is_complete("asr", root):
        raise model_error("MODEL_INSTALL_FAILED", "ASR model cache is incomplete after installation")
    create_manifest(
        root,
        "asr",
        [MODELSCOPE_SENSEVOICE],
        _artifact_paths("asr", root),
        requested_revision=MODELSCOPE_SENSEVOICE_REVISION,
    )
    return model_dir


def prepare_tts_model(
    *,
    validate: bool = True,
    cache_root: Path | None = None,
    allow_download: bool = False,
) -> tuple[Path, Path, Path]:
    root = cache_root or CACHE_ROOT
    if _is_complete("tts", root):
        _validate_default_prompt(root)
        _adopt_manifest_if_needed("tts", root)
        verify_manifest(root, "tts")
        paths = _tts_paths(root)
        if validate:
            _load_tts(paths[0])
        return paths

    if load_manifest(root, "tts") is not None:
        verify_manifest(root, "tts")

    _require_download_allowed("tts", cache_root=cache_root, explicit=allow_download)
    paths = _download_tts(root)
    _load_tts(paths[0])
    if not _is_complete("tts", root):
        raise model_error("MODEL_INSTALL_FAILED", "TTS model cache is incomplete after installation")
    create_manifest(
        root,
        "tts",
        [MODELSCOPE_MOSS_TTS, MODELSCOPE_MOSS_AUDIO_TOKENIZER],
        _artifact_paths("tts", root),
    )
    return paths


def _download_asr(root: Path) -> Path:
    from modelcli.models.cache import ensure_file_from_url, ensure_modelscope

    model_dir = asr_cache_dir(root)
    needs_download = not all(_is_nonempty_file(model_dir / name) for name in ASR_DOWNLOAD_FILES)
    if needs_download:
        (model_dir / ".download_ok").unlink(missing_ok=True)
    model_dir = ensure_modelscope(
        MODELSCOPE_SENSEVOICE,
        revision=MODELSCOPE_SENSEVOICE_REVISION,
        cache_root=root,
    )
    ensure_file_from_url(
        SENSEVOICE_TOKENIZER_URL,
        SENSEVOICE_TOKENIZER_NAME,
        model_dir,
        expected_sha256=SENSEVOICE_TOKENIZER_SHA256,
    )
    return model_dir


def _download_tts(root: Path) -> tuple[Path, Path, Path]:
    from modelcli.models.cache import ensure_file_from_url, ensure_modelscope

    prompt = ensure_file_from_url(
        MOSS_DEFAULT_PROMPT_URL,
        MOSS_DEFAULT_PROMPT_NAME,
        root / "moss_prompts",
        expected_sha256=MOSS_DEFAULT_PROMPT_SHA256,
    )
    main_dir = ensure_modelscope(MODELSCOPE_MOSS_TTS, cache_root=root)
    tokenizer_dir = ensure_modelscope(MODELSCOPE_MOSS_AUDIO_TOKENIZER, cache_root=root)
    return main_dir, tokenizer_dir, prompt


def _load_asr(model_dir: Path) -> None:
    from funasr_onnx import SenseVoiceSmall

    try:
        SenseVoiceSmall(str(model_dir), batch_size=1, quantize=True)
    except Exception as exc:
        raise model_error("MODEL_LOAD_FAILED", f"Failed to load ASR model: {exc}") from exc


def _load_tts(main_dir: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(str(main_dir), trust_remote_code=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        raise model_error("MODEL_LOAD_FAILED", f"Failed to load TTS model: {exc}") from exc


def _refresh_model(name: str) -> None:
    CACHE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".modelcli-{name}-refresh-", dir=CACHE_ROOT.parent))
    try:
        if name == "asr":
            prepare_asr_model(validate=True, cache_root=temporary)
        else:
            prepare_tts_model(validate=True, cache_root=temporary)
        verify_manifest(temporary, name)
        _publish_refresh(name, temporary)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _publish_refresh(name: str, source_root: Path) -> None:
    relatives = _owned_relatives(name)
    backup_root = Path(tempfile.mkdtemp(prefix=f".modelcli-{name}-backup-", dir=CACHE_ROOT.parent))
    published: list[Path] = []
    backed_up: list[tuple[Path, Path]] = []
    try:
        for relative in relatives:
            source = source_root / relative
            destination = CACHE_ROOT / relative
            backup = backup_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
                backed_up.append((backup, destination))
            if source.exists():
                os.replace(source, destination)
                published.append(destination)
        if not _is_complete(name, CACHE_ROOT):
            raise model_error("MODEL_INSTALL_FAILED", f"Refreshed model '{name}' is incomplete")
        verify_manifest(CACHE_ROOT, name)
    except Exception:
        for destination in reversed(published):
            _remove_path(destination)
        for backup, destination in reversed(backed_up):
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def _remove_one(name: str) -> bool:
    changed = False
    for relative in _owned_relatives(name):
        path = CACHE_ROOT / relative
        if path.exists():
            _remove_path(path)
            changed = True
    if name == "tts":
        prompt_dir = CACHE_ROOT / "moss_prompts"
        if prompt_dir.exists() and not any(prompt_dir.iterdir()):
            prompt_dir.rmdir()
    if name == "asr":
        _remove_modelscope_locks()
    return changed


def _managed_status(name: str) -> ModelStatus:
    complete = _is_complete(name, CACHE_ROOT)
    manifest_state = "missing"
    requested_revision = None
    source_revision = None
    try:
        manifest = load_manifest(CACHE_ROOT, name)
        if manifest is not None:
            manifest_state = "present" if _manifest_matches_standard(name, manifest) else "stale"
            requested_revision = manifest.get("requested_revision")
            source_revision = manifest.get("source_revision")
    except Exception:
        manifest_state = "invalid"
    model_name = ASR_MODEL_NAME if name == "asr" else TTS_MODEL_NAME
    return ModelStatus(
        name=name,
        model=model_name,
        status="installed" if complete else "missing",
        size_bytes=_owned_size(name) if complete else 0,
        managed=True,
        manifest_status=manifest_state,
        verification_status="not_checked" if complete else "not_verified",
        requested_revision=requested_revision,
        source_revision=source_revision,
    )


def _adopt_manifest_if_needed(name: str, root: Path) -> None:
    manifest = load_manifest(root, name)
    if manifest is not None and _manifest_matches_standard(name, manifest):
        return
    if name == "tts":
        _validate_default_prompt(root)
    model_ids = (
        [MODELSCOPE_SENSEVOICE]
        if name == "asr"
        else [MODELSCOPE_MOSS_TTS, MODELSCOPE_MOSS_AUDIO_TOKENIZER]
    )
    create_manifest(
        root,
        name,
        model_ids,
        _artifact_paths(name, root),
        requested_revision=_requested_revision(name),
    )


def _manifest_matches_standard(name: str, manifest: dict) -> bool:
    expected_ids = (
        [MODELSCOPE_SENSEVOICE]
        if name == "asr"
        else [MODELSCOPE_MOSS_TTS, MODELSCOPE_MOSS_AUDIO_TOKENIZER]
    )
    return (
        manifest.get("model_ids") == expected_ids
        and manifest.get("requested_revision") == _requested_revision(name)
    )


def _requested_revision(name: str) -> str:
    if name == "asr":
        return MODELSCOPE_SENSEVOICE_REVISION
    return MODELSCOPE_REVISION


def _artifact_paths(name: str, root: Path) -> list[Path]:
    if name == "asr":
        model_dir = asr_cache_dir(root)
        return [
            model_dir / filename
            for filename in ASR_REQUIRED_FILES
            if (model_dir / filename).is_file()
        ]
    main, tokenizer, prompt = _tts_paths(root)
    return [
        *[path for path in main.rglob("*") if path.is_file()],
        *[path for path in tokenizer.rglob("*") if path.is_file()],
        prompt,
    ]


def _is_complete(name: str, root: Path) -> bool:
    if name == "asr":
        model_dir = asr_cache_dir(root)
        if not all(_is_nonempty_file(model_dir / item) for item in ASR_REQUIRED_FILES):
            return False
        return sha256_file(model_dir / SENSEVOICE_TOKENIZER_NAME) == SENSEVOICE_TOKENIZER_SHA256
    return all(_is_nonempty_file(root / item) for item in TTS_REQUIRED_FILES)


def _validate_default_prompt(root: Path) -> None:
    prompt = root / "moss_prompts" / MOSS_DEFAULT_PROMPT_NAME
    if not prompt.is_file() or sha256_file(prompt) != MOSS_DEFAULT_PROMPT_SHA256:
        raise model_error("MODEL_VERIFICATION_FAILED", "Default TTS prompt hash mismatch")


def _require_download_allowed(
    name: str,
    *,
    cache_root: Path | None,
    explicit: bool,
) -> None:
    if cache_root is None and not explicit and not current_runtime().allow_download:
        raise model_error("MODEL_NOT_INSTALLED", f"Model '{name}' is not installed")


def _tts_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / MODELSCOPE_MOSS_TTS.replace("/", "__"),
        root / MODELSCOPE_MOSS_AUDIO_TOKENIZER.replace("/", "__"),
        root / "moss_prompts" / MOSS_DEFAULT_PROMPT_NAME,
    )


def _owned_relatives(name: str) -> tuple[Path, ...]:
    if name == "asr":
        return (
            Path(MODELSCOPE_SENSEVOICE.replace("/", "__")),
            Path("manifests/asr.json"),
        )
    return (
        Path(MODELSCOPE_MOSS_TTS.replace("/", "__")),
        Path(MODELSCOPE_MOSS_AUDIO_TOKENIZER.replace("/", "__")),
        Path("moss_prompts") / MOSS_DEFAULT_PROMPT_NAME,
        Path("manifests/tts.json"),
    )


def _target_names(target: ModelTarget) -> tuple[str, ...]:
    return ("asr", "tts") if target == ModelTarget.all else (target.value,)


def _owned_size(name: str) -> int:
    return sum(_path_size(CACHE_ROOT / relative) for relative in _owned_relatives(name))


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file() and not child.is_symlink())


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _remove_modelscope_locks() -> None:
    lock_dir = CACHE_ROOT / ".lock"
    if not lock_dir.exists():
        return
    model_key = MODELSCOPE_SENSEVOICE.replace("/", "___")
    for lock_file in lock_dir.glob(f"model_{model_key}*"):
        if lock_file.is_file():
            lock_file.unlink()
