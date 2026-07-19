"""Filesystem publication helpers for managed model caches."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path


def publish_refresh(
    *,
    name: str,
    source_root: Path,
    cache_root: Path,
    relatives: tuple[Path, ...],
    validate: Callable[[], None],
) -> None:
    backup_root = Path(
        tempfile.mkdtemp(prefix=f".modelcli-{name}-backup-", dir=cache_root.parent)
    )
    published: list[Path] = []
    backed_up: list[tuple[Path, Path]] = []
    try:
        for relative in relatives:
            source = source_root / relative
            destination = cache_root / relative
            backup = backup_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
                backed_up.append((backup, destination))
            if source.exists():
                os.replace(source, destination)
                published.append(destination)
        validate()
    except Exception:
        for destination in reversed(published):
            remove_path(destination)
        for backup, destination in reversed(backed_up):
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file() and not child.is_symlink()
    )


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
