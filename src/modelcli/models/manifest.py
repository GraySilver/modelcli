"""Local artifact manifests for managed model caches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelcli import __version__
from modelcli.config import MODELSCOPE_REVISION
from modelcli.errors import model_error

MANIFEST_SCHEMA_VERSION = "1"


def manifest_path(cache_root: Path, name: str) -> Path:
    return cache_root / "manifests" / f"{name}.json"


def create_manifest(
    cache_root: Path,
    name: str,
    model_ids: list[str],
    artifact_paths: list[Path],
    *,
    requested_revision: str = MODELSCOPE_REVISION,
) -> dict[str, Any]:
    files: list[dict[str, str | int]] = []
    for path in sorted(set(artifact_paths)):
        if not path.is_file() or path.is_symlink() or path.name == ".download_ok":
            continue
        try:
            relative = path.relative_to(cache_root)
        except ValueError as exc:
            raise model_error("MODEL_VERIFICATION_FAILED", f"Artifact is outside cache: {path}") from exc
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    if not files:
        raise model_error("MODEL_VERIFICATION_FAILED", f"No artifacts found for model '{name}'")

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": name,
        "model_ids": model_ids,
        "requested_revision": requested_revision,
        "source_revision": None,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "modelcli_version": __version__,
        "files": files,
    }
    write_manifest(cache_root, name, manifest)
    return manifest


def write_manifest(cache_root: Path, name: str, manifest: dict[str, Any]) -> None:
    destination = manifest_path(cache_root, name)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".json.tmp", dir=destination.parent
        )
        temporary = Path(raw_path)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    except OSError as exc:
        raise model_error(
            "MODEL_VERIFICATION_FAILED",
            f"Cannot write manifest for model '{name}'",
        ) from exc
    finally:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)


def load_manifest(cache_root: Path, name: str) -> dict[str, Any] | None:
    path = manifest_path(cache_root, name)
    if not path.is_file():
        return None
    if path.is_symlink():
        raise model_error("MODEL_VERIFICATION_FAILED", f"Invalid manifest for model '{name}'")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise model_error("MODEL_VERIFICATION_FAILED", f"Invalid manifest for model '{name}'") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or value.get("name") != name
        or not isinstance(value.get("files"), list)
    ):
        raise model_error("MODEL_VERIFICATION_FAILED", f"Invalid manifest for model '{name}'")
    return value


def verify_manifest(cache_root: Path, name: str) -> dict[str, Any]:
    manifest = load_manifest(cache_root, name)
    if manifest is None:
        raise model_error("MODEL_MANIFEST_MISSING", f"Manifest is missing for model '{name}'")

    checked = 0
    size_bytes = 0
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise model_error("MODEL_VERIFICATION_FAILED", f"Invalid manifest entry for model '{name}'")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise model_error("MODEL_VERIFICATION_FAILED", f"Invalid manifest entry for model '{name}'")
        path = cache_root / relative
        expected_size = entry.get("size_bytes")
        expected_hash = entry.get("sha256")
        if not path.is_file() or path.is_symlink():
            raise model_error("MODEL_VERIFICATION_FAILED", f"Model artifact is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size or sha256_file(path) != expected_hash:
            raise model_error("MODEL_VERIFICATION_FAILED", f"Model artifact hash mismatch: {path}")
        checked += 1
        size_bytes += actual_size

    return {
        "name": name,
        "verified": True,
        "files_checked": checked,
        "size_bytes": size_bytes,
        "requested_revision": manifest.get("requested_revision"),
        "source_revision": manifest.get("source_revision"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise model_error("MODEL_VERIFICATION_FAILED", f"Cannot read model artifact: {path}") from exc
    return digest.hexdigest()
