"""Managed lifecycle for the fixed PicoDet detection model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modelcli.config import (
    CACHE_ROOT,
    PICODET_CACHE_NAME,
    PICODET_MODEL_ID,
    PICODET_MODEL_NAME,
    PICODET_MODEL_REVISION,
    PICODET_MODEL_SHA256,
    PICODET_MODEL_SIZE,
    PICODET_MODEL_URL,
)
from modelcli.errors import model_error
from modelcli.models.manifest import create_manifest, load_manifest, sha256_file, verify_manifest
from modelcli.protocol import current_runtime

DETECT_MODEL_NAME = "PicoDet-L 416 COCO"
DETECT_REQUIRED_FILES = (PICODET_MODEL_NAME,)


def detect_cache_dir(cache_root: Path | None = None) -> Path:
    return (cache_root or CACHE_ROOT) / PICODET_CACHE_NAME


def prepare_detect_model(
    *,
    validate: bool,
    cache_root: Path | None = None,
    allow_download: bool = False,
) -> Path:
    root = cache_root or CACHE_ROOT
    model_dir = detect_cache_dir(root)
    manifest = load_manifest(root, "detect")
    if is_detect_complete(root):
        loaded = False
        if manifest is None or not detect_manifest_matches(manifest):
            load_detect_model(model_dir)
            loaded = True
            create_detect_manifest(root)
        verify_manifest(root, "detect")
        if validate and not loaded:
            load_detect_model(model_dir)
        return model_dir

    if manifest is not None and detect_manifest_matches(manifest):
        verify_manifest(root, "detect")
    _require_download_allowed(cache_root=cache_root, explicit=allow_download)
    model_dir = download_detect_model(root)
    load_detect_model(model_dir)
    if not is_detect_complete(root):
        raise model_error(
            "MODEL_INSTALL_FAILED",
            "Detection model cache is incomplete after installation",
        )
    create_detect_manifest(root)
    return model_dir


def download_detect_model(root: Path) -> Path:
    from modelcli.models.cache import ensure_file_from_url

    model_dir = detect_cache_dir(root)
    ensure_file_from_url(
        PICODET_MODEL_URL,
        PICODET_MODEL_NAME,
        model_dir,
        expected_sha256=PICODET_MODEL_SHA256,
    )
    return model_dir


def load_detect_model(model_dir: Path) -> None:
    from modelcli.detect.engine import load_detection_session

    load_detection_session(model_dir / PICODET_MODEL_NAME)


def create_detect_manifest(root: Path) -> dict[str, Any]:
    return create_manifest(
        root,
        "detect",
        [PICODET_MODEL_ID],
        detect_artifact_paths(root),
        requested_revision=PICODET_MODEL_REVISION,
    )


def detect_manifest_matches(manifest: dict[str, Any]) -> bool:
    return (
        manifest.get("model_ids") == [PICODET_MODEL_ID]
        and manifest.get("requested_revision") == PICODET_MODEL_REVISION
    )


def detect_artifact_paths(root: Path) -> list[Path]:
    return [detect_cache_dir(root) / PICODET_MODEL_NAME]


def is_detect_complete(root: Path) -> bool:
    model = detect_cache_dir(root) / PICODET_MODEL_NAME
    return (
        model.is_file()
        and not model.is_symlink()
        and model.stat().st_size == PICODET_MODEL_SIZE
        and sha256_file(model) == PICODET_MODEL_SHA256
    )


def detect_owned_relatives() -> tuple[Path, ...]:
    return (Path(PICODET_CACHE_NAME), Path("manifests/detect.json"))


def _require_download_allowed(*, cache_root: Path | None, explicit: bool) -> None:
    if cache_root is None and not explicit and not current_runtime().allow_download:
        raise model_error("MODEL_NOT_INSTALLED", "Model 'detect' is not installed")
