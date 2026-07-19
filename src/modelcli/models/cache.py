"""Model download / cache helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import urlretrieve

from rich.console import Console

from modelcli.config import CACHE_ROOT

console = Console(stderr=True)


def model_dir(model_id: str) -> Path:
    """Return the local cache directory for a ModelScope model id (may not exist)."""
    return CACHE_ROOT / model_id.replace("/", "__")


def is_installed(model_id: str) -> bool:
    d = model_dir(model_id)
    return (d / ".download_ok").exists()


def ensure_modelscope(model_id: str, revision: str | None = None) -> Path:
    """Download (if needed) a ModelScope model and return its local directory."""
    from modelscope.hub.snapshot_download import snapshot_download

    local_dir = model_dir(model_id)
    marker = local_dir / ".download_ok"
    if marker.exists():
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    with console.status(f"[bold cyan]Downloading model[/bold cyan] {model_id} ..."):
        snapshot_download(
            model_id,
            revision=revision,
            cache_dir=str(CACHE_ROOT),
            local_dir=str(local_dir),
        )
    marker.write_text("ok")
    return local_dir


def ensure_file_from_url(url: str, filename: str, dest_dir: Path) -> Path:
    """Download a single file into dest_dir/filename (skip if it already exists)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with console.status(f"[bold cyan]Downloading[/bold cyan] {filename} ..."):
        urlretrieve(url, tmp)
    tmp.rename(dest)
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
