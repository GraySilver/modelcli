"""Cross-process locks for each managed model capability."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filelock import FileLock, Timeout

from modelcli.config import MODEL_LOCK_TIMEOUT_SECONDS
from modelcli.errors import model_error


@contextmanager
def model_lock(
    name: str,
    cache_root: Path,
    *,
    timeout: float = MODEL_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    lock_path = cache_root / ".modelcli-locks" / f"{name}.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(lock_path, timeout=timeout)
        lock.acquire()
    except Timeout as exc:
        raise model_error(
            "MODEL_BUSY",
            f"Model '{name}' is busy; lock wait exceeded {timeout:g} seconds",
            retryable=True,
        ) from exc
    except OSError as exc:
        raise model_error(
            "MODEL_LOCK_FAILED",
            f"Cannot lock model '{name}': {exc}",
            retryable=True,
        ) from exc

    try:
        yield
    finally:
        lock.release()
