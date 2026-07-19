"""Stable errors exposed by the command-line contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ExitCode(IntEnum):
    SUCCESS = 0
    USAGE = 2
    INVALID_INPUT = 3
    MODEL = 4
    INFERENCE = 5
    OUTPUT = 6
    INTERRUPTED = 130


@dataclass(eq=False)
class ModelCliError(Exception):
    code: str
    message: str
    exit_code: ExitCode
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


def invalid_input(code: str, message: str) -> ModelCliError:
    return ModelCliError(code, message, ExitCode.INVALID_INPUT)


def model_error(code: str, message: str, *, retryable: bool = False) -> ModelCliError:
    return ModelCliError(code, message, ExitCode.MODEL, retryable)


def inference_error(code: str, message: str, *, retryable: bool = False) -> ModelCliError:
    return ModelCliError(code, message, ExitCode.INFERENCE, retryable)


def output_error(code: str, message: str) -> ModelCliError:
    return ModelCliError(code, message, ExitCode.OUTPUT)
