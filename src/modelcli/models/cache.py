"""Model download / cache helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from urllib.request import urlopen

from rich.console import Console

from modelcli.config import CACHE_ROOT, DOWNLOAD_TIMEOUT_SECONDS, MODELSCOPE_REVISION
from modelcli.errors import model_error

console = Console(stderr=True)


def model_dir(model_id: str, *, cache_root: Path | None = None) -> Path:
    """Return the local cache directory for a ModelScope model id (may not exist)."""
    return (cache_root or CACHE_ROOT) / model_id.replace("/", "__")


def is_installed(model_id: str) -> bool:
    d = model_dir(model_id)
    return (d / ".download_ok").exists()


def ensure_modelscope(
    model_id: str,
    revision: str = MODELSCOPE_REVISION,
    *,
    cache_root: Path | None = None,
) -> Path:
    """Download (if needed) a ModelScope model and return its local directory."""
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_TIMEOUT", str(DOWNLOAD_TIMEOUT_SECONDS))
    from modelscope.hub.snapshot_download import snapshot_download

    root = cache_root or CACHE_ROOT
    local_dir = model_dir(model_id, cache_root=root)
    marker = local_dir / ".download_ok"
    if marker.exists():
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    with console.status(f"[bold cyan]Downloading model[/bold cyan] {model_id} ..."):
        try:
            snapshot_download(
                model_id,
                revision=revision,
                cache_dir=str(root),
                local_dir=str(local_dir),
            )
        except Exception as exc:
            raise model_error(
                "MODEL_DOWNLOAD_FAILED",
                f"Failed to download model '{model_id}': {exc}",
                retryable=True,
            ) from exc
    marker.write_text("ok", encoding="utf-8")
    return local_dir


def ensure_file_from_url(
    url: str,
    filename: str,
    dest_dir: Path,
    *,
    expected_sha256: str,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """Download a single file into dest_dir/filename (skip if it already exists)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists() and _sha256(dest) == expected_sha256:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with console.status(f"[bold cyan]Downloading[/bold cyan] {filename} ..."):
        try:
            with urlopen(url, timeout=timeout) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise model_error(
                "MODEL_DOWNLOAD_FAILED",
                f"Failed to download '{filename}': {exc}",
                retryable=True,
            ) from exc
    if _sha256(tmp) != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise model_error("MODEL_VERIFICATION_FAILED", f"Hash mismatch for downloaded file: {filename}")
    os.replace(tmp, dest)
    return dest


def dir_size(path: Path) -> int:
    """Return total size of all files under a directory in bytes."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def remove_model(model_id: str) -> bool:
    """Delete a cached model. Returns True if something was deleted."""
    d = model_dir(model_id)
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
