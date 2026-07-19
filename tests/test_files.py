from pathlib import Path

import pytest

from modelcli.errors import ModelCliError
from modelcli.files import atomic_output_path


def test_atomic_output_replaces_only_with_force_and_cleans_temp(tmp_path: Path) -> None:
    destination = tmp_path / "result.wav"
    destination.write_bytes(b"old")

    with pytest.raises(ModelCliError) as raised:
        with atomic_output_path(destination, force=False):
            pass
    assert raised.value.code == "OUTPUT_EXISTS"

    with atomic_output_path(destination, force=True) as temporary:
        temporary.write_bytes(b"new")

    assert destination.read_bytes() == b"new"
    assert list(tmp_path.glob(".result.wav.*")) == []


def test_atomic_output_preserves_destination_on_failure(tmp_path: Path) -> None:
    destination = tmp_path / "result.png"
    destination.write_bytes(b"old")

    with pytest.raises(RuntimeError):
        with atomic_output_path(destination, force=True) as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("failed")

    assert destination.read_bytes() == b"old"
    assert list(tmp_path.glob(".result.png.*")) == []


def test_atomic_output_does_not_reclassify_body_oserror(tmp_path: Path) -> None:
    destination = tmp_path / "result.wav"

    with pytest.raises(OSError, match="inference read failed"):
        with atomic_output_path(destination, force=False):
            raise OSError("inference read failed")

    assert not destination.exists()
