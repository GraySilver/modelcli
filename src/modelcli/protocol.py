"""Runtime context and the ModelCLI Agent JSON envelope."""

from __future__ import annotations

import json
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from modelcli import __version__
from modelcli.errors import ModelCliError

SCHEMA_VERSION = "1"


@dataclass
class RuntimeContext:
    json_mode: bool = False
    allow_download: bool = True
    debug: bool = False
    operation: str = "modelcli"
    started_at: float = field(default_factory=time.monotonic)
    emitted: bool = False


_CURRENT_RUNTIME: ContextVar[RuntimeContext] = ContextVar(
    "modelcli_runtime",
    default=RuntimeContext(),
)


def current_runtime() -> RuntimeContext:
    return _CURRENT_RUNTIME.get()


def set_runtime(runtime: RuntimeContext) -> None:
    _CURRENT_RUNTIME.set(runtime)


def success(result: dict[str, Any], *, operation: str | None = None) -> None:
    runtime = current_runtime()
    if not runtime.json_mode:
        return
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "operation": operation or runtime.operation,
            "result": result,
            "meta": _meta(runtime),
        }
    )


def failure(error: ModelCliError, *, operation: str | None = None) -> None:
    runtime = current_runtime()
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "operation": operation or runtime.operation,
            "error": {
                "code": error.code,
                "message": error.message,
                "retryable": error.retryable,
            },
            "meta": _meta(runtime),
        }
    )


def _meta(runtime: RuntimeContext) -> dict[str, str | int]:
    return {
        "modelcli_version": __version__,
        "elapsed_ms": max(0, round((time.monotonic() - runtime.started_at) * 1000)),
    }


def _emit(payload: dict[str, Any]) -> None:
    runtime = current_runtime()
    if runtime.emitted:
        return
    runtime.emitted = True
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
