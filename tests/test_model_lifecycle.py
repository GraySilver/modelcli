from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modelcli.cli import app
from modelcli.models import lifecycle

runner = CliRunner()


def _write_asr_cache(root: Path) -> Path:
    model_dir = root / "iic__SenseVoiceSmall"
    for relative in lifecycle.ASR_REQUIRED_FILES:
        path = model_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return model_dir


def _write_tts_files(root: Path) -> dict[str, str]:
    """Write MOSS TTS sentinel files (main model, tokenizer, default prompt)."""
    paths: dict[str, str] = {}
    for relative in lifecycle.TTS_REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        paths[relative] = str(path)
    return paths


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(lifecycle, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr("modelcli.models.cache.CACHE_ROOT", tmp_path)
    return tmp_path


def test_list_models_has_fixed_managed_and_bundled_capabilities(
    isolated_cache: Path,
) -> None:
    asr_dir = _write_asr_cache(isolated_cache)
    (asr_dir / "weights-link").symlink_to(asr_dir / "model.onnx")
    _write_tts_files(isolated_cache)

    models = lifecycle.list_models()

    assert [(model.name, model.status, model.managed) for model in models] == [
        ("asr", "installed", True),
        ("tts", "installed", True),
        ("ocr", "bundled", False),
        ("vad", "bundled", False),
    ]
    assert models[0].size_bytes == len(lifecycle.ASR_REQUIRED_FILES)


def test_missing_required_file_marks_model_missing(isolated_cache: Path) -> None:
    model_dir = _write_asr_cache(isolated_cache)
    (model_dir / "tokens.json").write_bytes(b"")

    assert lifecycle.get_model_status("asr").status == "missing"


def test_tts_mirror_file_layout_is_installed(isolated_cache: Path) -> None:
    _write_tts_files(isolated_cache)

    assert lifecycle.get_model_status("tts").status == "installed"


def test_install_is_idempotent_but_always_validates(
    monkeypatch,
    isolated_cache: Path,
) -> None:
    validations: list[bool] = []

    def prepare(*, validate: bool) -> Path:
        validations.append(validate)
        return _write_asr_cache(isolated_cache)

    monkeypatch.setattr(lifecycle, "prepare_asr_model", prepare)

    first = lifecycle.install_models(lifecycle.ModelTarget.asr)
    second = lifecycle.install_models(lifecycle.ModelTarget.asr)

    assert first[0].changed is True
    assert second[0].changed is False
    assert validations == [True, True]


def test_prepare_asr_repairs_missing_download_and_onnx_files(
    monkeypatch,
    isolated_cache: Path,
) -> None:
    model_dir = _write_asr_cache(isolated_cache)
    (model_dir / "tokens.json").unlink()
    calls: list[str] = []

    def ensure(_model_id: str) -> Path:
        calls.append("download")
        for relative in lifecycle.ASR_DOWNLOAD_FILES:
            path = model_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"source")
        (model_dir / ".download_ok").write_bytes(b"ok")
        return model_dir

    class FakeSenseVoice:
        def __init__(self, path: str, batch_size: int) -> None:
            assert path == str(model_dir)
            assert batch_size == 1
            calls.append("validate")
            (model_dir / "model.onnx").write_bytes(b"onnx")

    monkeypatch.setattr("modelcli.models.cache.ensure_modelscope", ensure)
    monkeypatch.setattr("funasr_onnx.SenseVoiceSmall", FakeSenseVoice)

    lifecycle.prepare_asr_model(validate=False)

    assert calls == ["download", "validate"]
    assert lifecycle.get_model_status("asr").status == "installed"


def test_install_all_keeps_successful_asr_when_tts_fails(
    monkeypatch,
    isolated_cache: Path,
) -> None:
    monkeypatch.setattr(
        lifecycle,
        "prepare_asr_model",
        lambda **_kwargs: _write_asr_cache(isolated_cache),
    )
    monkeypatch.setattr(
        lifecycle,
        "prepare_tts_model",
        lambda: (_ for _ in ()).throw(RuntimeError("tts unavailable")),
    )

    with pytest.raises(RuntimeError, match="tts unavailable"):
        lifecycle.install_models(lifecycle.ModelTarget.all)

    assert lifecycle.get_model_status("asr").status == "installed"


def test_remove_is_targeted_idempotent_and_cleans_asr_locks(
    isolated_cache: Path,
) -> None:
    asr_dir = _write_asr_cache(isolated_cache)
    tts_files = _write_tts_files(isolated_cache)
    unrelated = isolated_cache / "unmanaged.bin"
    unrelated.write_bytes(b"keep")
    shared_hf = isolated_cache.parent / "shared-huggingface"
    shared_hf.mkdir()
    (shared_hf / "keep.bin").write_bytes(b"keep")
    lock_dir = isolated_cache / ".lock"
    lock_dir.mkdir()
    matching_lock = lock_dir / "model_iic___SenseVoiceSmall_model_pt.lock"
    matching_lock.write_bytes(b"")
    other_lock = lock_dir / "model_other_repo.lock"
    other_lock.write_bytes(b"")

    first = lifecycle.remove_models(lifecycle.ModelTarget.asr)
    second = lifecycle.remove_models(lifecycle.ModelTarget.asr)

    assert first[0].changed is True
    assert second[0].changed is False
    assert not asr_dir.exists()
    # TTS sentinel files must remain untouched.
    for rel in lifecycle.TTS_REQUIRED_FILES:
        assert (isolated_cache / rel).exists(), rel
    assert unrelated.exists()
    assert (shared_hf / "keep.bin").exists()
    assert not matching_lock.exists()
    assert other_lock.exists()


def test_remove_all_only_removes_managed_models(isolated_cache: Path) -> None:
    _write_asr_cache(isolated_cache)
    _write_tts_files(isolated_cache)
    bundled_artifact = isolated_cache / "silero_vad.onnx"
    bundled_artifact.write_bytes(b"legacy")

    results = lifecycle.remove_models(lifecycle.ModelTarget.all)

    assert [result.name for result in results] == ["asr", "tts"]
    assert bundled_artifact.exists()


def test_models_list_json_has_stable_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        lifecycle,
        "list_models",
        lambda: [
            lifecycle.ModelStatus("asr", "SenseVoiceSmall", "missing", 0, True),
            lifecycle.ModelStatus("ocr", "PP-OCRv4 mobile", "bundled", 0, False),
        ],
    )

    result = runner.invoke(app, ["models", "list", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "models": [
            {
                "name": "asr",
                "model": "SenseVoiceSmall",
                "status": "missing",
                "size_bytes": 0,
                "managed": True,
            },
            {
                "name": "ocr",
                "model": "PP-OCRv4 mobile",
                "status": "bundled",
                "size_bytes": 0,
                "managed": False,
            },
        ]
    }


@pytest.mark.parametrize(
    ("command", "action"),
    [
        (["models", "install", "asr", "--json"], "install"),
        (["models", "remove", "tts", "--json"], "remove"),
        (["models", "prefetch", "--json"], "install"),
        (["models", "clean", "--json"], "remove"),
    ],
)
def test_model_action_json_and_legacy_delegation(
    monkeypatch,
    command: list[str],
    action: str,
) -> None:
    calls: list[lifecycle.ModelTarget] = []

    def run(target: lifecycle.ModelTarget) -> list[lifecycle.ModelActionResult]:
        calls.append(target)
        name = "asr" if target in (lifecycle.ModelTarget.asr, lifecycle.ModelTarget.all) else "tts"
        status = "installed" if action == "install" else "missing"
        return [lifecycle.ModelActionResult(name, status, True, 123)]

    function = "install_models" if action == "install" else "remove_models"
    monkeypatch.setattr(lifecycle, function, run)

    result = runner.invoke(app, command)

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["action"] == action
    assert payload["target"] == calls[0].value
    if command[1] in ("prefetch", "clean"):
        assert calls == [lifecycle.ModelTarget.all]


def test_invalid_model_target_is_usage_error() -> None:
    result = runner.invoke(app, ["models", "install", "ocr"])

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_action_failure_is_concise(monkeypatch) -> None:
    monkeypatch.setattr(
        lifecycle,
        "install_models",
        lambda _target: (_ for _ in ()).throw(RuntimeError("download failed")),
    )

    result = runner.invoke(app, ["models", "install", "asr", "--json"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "download failed" in result.stderr
    assert "Traceback" not in result.output
