"""Data contracts for managed model lifecycle operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal


class ModelTarget(str, Enum):
    detect = "detect"
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
