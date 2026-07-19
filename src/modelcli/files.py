"""Predictable file-output helpers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from modelcli.errors import ModelCliError, output_error


def ensure_output_available(destination: Path, *, force: bool) -> None:
    if destination.expanduser().exists() and not force:
        raise output_error("OUTPUT_EXISTS", f"Output already exists: {destination}")


def write_text_output(destination: Path, value: str) -> None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(value, encoding="utf-8")
    except OSError as exc:
        raise output_error("OUTPUT_WRITE_FAILED", f"Cannot write output: {destination}") from exc


@contextmanager
def atomic_output_path(destination: Path, *, force: bool) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically publish it on success."""
    destination = destination.expanduser()
    ensure_output_available(destination, force=force)

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = f".tmp{destination.suffix}" if destination.suffix else ".tmp"
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=suffix,
            dir=destination.parent,
        )
        os.close(fd)
        temporary = Path(raw_path)
    except OSError as exc:
        raise output_error("OUTPUT_WRITE_FAILED", f"Cannot prepare output: {destination}") from exc

    try:
        yield temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    try:
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise output_error("OUTPUT_WRITE_FAILED", f"Output was not written: {destination}")
        os.replace(temporary, destination)
    except ModelCliError:
        raise
    except OSError as exc:
        raise output_error("OUTPUT_WRITE_FAILED", f"Cannot write output: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)
