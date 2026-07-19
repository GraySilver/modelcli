from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from modelcli.errors import ModelCliError
from modelcli.models import detection, lifecycle
from modelcli.models.manifest import manifest_path, verify_manifest
from modelcli.protocol import RuntimeContext, set_runtime


@pytest.fixture
def detection_cache(monkeypatch, tmp_path: Path) -> Path:
    content = b"fixed-picodet"
    monkeypatch.setattr(detection, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(lifecycle, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr("modelcli.models.cache.CACHE_ROOT", tmp_path)
    monkeypatch.setattr(detection, "PICODET_MODEL_SIZE", len(content))
    monkeypatch.setattr(detection, "PICODET_MODEL_SHA256", hashlib.sha256(content).hexdigest())
    set_runtime(RuntimeContext())
    return tmp_path


def _write_model(root: Path, content: bytes = b"fixed-picodet") -> Path:
    model_dir = detection.detect_cache_dir(root)
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / detection.PICODET_MODEL_NAME
    model.write_bytes(content)
    return model_dir


def test_complete_detection_cache_loads_before_manifest_adoption(
    monkeypatch,
    detection_cache: Path,
) -> None:
    model_dir = _write_model(detection_cache)
    calls: list[Path] = []
    monkeypatch.setattr(detection, "load_detect_model", calls.append)

    result = detection.prepare_detect_model(validate=False)

    assert result == model_dir
    assert calls == [model_dir]
    verified = verify_manifest(detection_cache, "detect")
    assert verified["requested_revision"] == "release/2.8"
    assert verified["files_checked"] == 1


def test_agent_detection_blocks_implicit_download(detection_cache: Path) -> None:
    set_runtime(RuntimeContext(json_mode=True, allow_download=False))

    with pytest.raises(ModelCliError) as raised:
        detection.prepare_detect_model(validate=False)

    assert raised.value.code == "MODEL_NOT_INSTALLED"


def test_detection_download_uses_fixed_url_name_and_hash(monkeypatch, detection_cache: Path) -> None:
    calls: list[tuple] = []

    def ensure(url: str, filename: str, destination: Path, *, expected_sha256: str) -> Path:
        calls.append((url, filename, destination, expected_sha256))
        destination.mkdir(parents=True)
        model = destination / filename
        model.write_bytes(b"fixed-picodet")
        return model

    monkeypatch.setattr("modelcli.models.cache.ensure_file_from_url", ensure)

    result = detection.download_detect_model(detection_cache)

    assert result == detection.detect_cache_dir(detection_cache)
    assert calls == [
        (
            detection.PICODET_MODEL_URL,
            detection.PICODET_MODEL_NAME,
            detection.detect_cache_dir(detection_cache),
            detection.PICODET_MODEL_SHA256,
        )
    ]


def test_detect_install_and_remove_are_verified_targeted_and_idempotent(
    monkeypatch,
    detection_cache: Path,
) -> None:
    def prepare(*, validate: bool, allow_download: bool) -> Path:
        assert validate is True
        assert allow_download is True
        model_dir = _write_model(detection_cache)
        detection.create_detect_manifest(detection_cache)
        return model_dir

    monkeypatch.setattr(lifecycle, "prepare_detect_model", prepare)
    unrelated = detection_cache / "keep.bin"
    unrelated.write_bytes(b"keep")

    installed = lifecycle.install_models(lifecycle.ModelTarget.detect)
    removed = lifecycle.remove_models(lifecycle.ModelTarget.detect)
    removed_again = lifecycle.remove_models(lifecycle.ModelTarget.detect)

    assert installed[0].verification_status == "verified"
    assert installed[0].requested_revision == "release/2.8"
    assert removed[0].changed is True
    assert removed_again[0].changed is False
    assert unrelated.read_bytes() == b"keep"
    assert not manifest_path(detection_cache, "detect").exists()


def test_all_target_order_includes_detection() -> None:
    assert lifecycle._target_names(lifecycle.ModelTarget.all) == ("detect", "asr", "tts")
