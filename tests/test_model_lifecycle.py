from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from filelock import FileLock

from modelcli.errors import ModelCliError
from modelcli.models import lifecycle
from modelcli.models.locking import model_lock
from modelcli.models.manifest import create_manifest, manifest_path, verify_manifest
from modelcli.protocol import RuntimeContext, set_runtime


def _write_asr_cache(root: Path, content: bytes = b"x") -> Path:
    model_dir = root / lifecycle.MODELSCOPE_SENSEVOICE.replace("/", "__")
    for relative in lifecycle.ASR_REQUIRED_FILES:
        path = model_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tokenizer" if relative == lifecycle.SENSEVOICE_TOKENIZER_NAME else content)
    return model_dir


def _create_asr_manifest(root: Path, model_dir: Path) -> None:
    create_manifest(
        root,
        "asr",
        [lifecycle.MODELSCOPE_SENSEVOICE],
        [path for path in model_dir.iterdir() if path.name != ".download_ok"],
        requested_revision=lifecycle.MODELSCOPE_SENSEVOICE_REVISION,
    )


def _write_tts_cache(root: Path, prompt: bytes = b"prompt") -> None:
    for relative in lifecycle.TTS_REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(prompt if path.name == lifecycle.MOSS_DEFAULT_PROMPT_NAME else b"model")


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(lifecycle, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(
        lifecycle,
        "SENSEVOICE_TOKENIZER_SHA256",
        hashlib.sha256(b"tokenizer").hexdigest(),
    )
    monkeypatch.setattr("modelcli.models.cache.CACHE_ROOT", tmp_path)
    set_runtime(RuntimeContext())
    return tmp_path


def test_complete_existing_cache_is_adopted_without_download(monkeypatch, isolated_cache: Path) -> None:
    model_dir = _write_asr_cache(isolated_cache)
    monkeypatch.setattr(lifecycle, "_load_asr", lambda path: None)
    monkeypatch.setattr(
        "modelcli.models.cache.ensure_modelscope",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not download")),
    )

    result = lifecycle.prepare_asr_model(validate=True)

    assert result == model_dir
    assert manifest_path(isolated_cache, "asr").is_file()
    assert verify_manifest(isolated_cache, "asr")["verified"] is True


def test_manifest_detects_artifact_tampering(isolated_cache: Path) -> None:
    model_dir = _write_asr_cache(isolated_cache)
    _create_asr_manifest(isolated_cache, model_dir)
    (model_dir / "tokens.json").write_bytes(b"changed")

    with pytest.raises(ModelCliError) as raised:
        lifecycle.verify_models(lifecycle.ModelTarget.asr)

    assert raised.value.code == "MODEL_VERIFICATION_FAILED"


def test_manifest_rejects_paths_outside_cache(isolated_cache: Path) -> None:
    outside = isolated_cache.parent / "outside.bin"
    outside.write_bytes(b"outside")
    path = manifest_path(isolated_cache, "asr")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "name": "asr",
                "files": [
                    {
                        "path": "../outside.bin",
                        "size_bytes": outside.stat().st_size,
                        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )

    with pytest.raises(ModelCliError) as raised:
        verify_manifest(isolated_cache, "asr")

    assert raised.value.code == "MODEL_VERIFICATION_FAILED"


def test_agent_inference_blocks_implicit_model_download(isolated_cache: Path) -> None:
    set_runtime(RuntimeContext(json_mode=True, allow_download=False))

    with pytest.raises(ModelCliError) as raised:
        lifecycle.prepare_asr_model(validate=False)

    assert raised.value.code == "MODEL_NOT_INSTALLED"


def test_explicit_install_allows_download_in_agent_mode(monkeypatch, isolated_cache: Path) -> None:
    set_runtime(RuntimeContext(json_mode=True, allow_download=False))

    def prepare(*, validate: bool, allow_download: bool) -> Path:
        assert validate is True
        assert allow_download is True
        model_dir = _write_asr_cache(isolated_cache)
        _create_asr_manifest(isolated_cache, model_dir)
        return model_dir

    monkeypatch.setattr(lifecycle, "prepare_asr_model", prepare)

    result = lifecycle.install_models(lifecycle.ModelTarget.asr)

    assert result[0].status == "installed"
    assert result[0].verification_status == "verified"


def test_remove_is_targeted_and_idempotent(monkeypatch, isolated_cache: Path) -> None:
    _write_asr_cache(isolated_cache)
    legacy_asr = isolated_cache / "iic__SenseVoiceSmall"
    legacy_asr.mkdir()
    (legacy_asr / "model.pt").write_bytes(b"legacy")
    prompt = b"prompt"
    _write_tts_cache(isolated_cache, prompt)
    monkeypatch.setattr(lifecycle, "MOSS_DEFAULT_PROMPT_SHA256", hashlib.sha256(prompt).hexdigest())
    lifecycle._adopt_manifest_if_needed("tts", isolated_cache)
    unrelated = isolated_cache / "keep.bin"
    unrelated.write_bytes(b"keep")

    first = lifecycle.remove_models(lifecycle.ModelTarget.asr)
    second = lifecycle.remove_models(lifecycle.ModelTarget.asr)

    assert first[0].changed is True
    assert second[0].changed is False
    assert lifecycle._is_complete("tts", isolated_cache)
    assert (legacy_asr / "model.pt").read_bytes() == b"legacy"
    assert unrelated.read_bytes() == b"keep"


def test_model_lock_timeout_has_stable_error(tmp_path: Path) -> None:
    lock_path = tmp_path / ".modelcli-locks" / "asr.lock"
    lock_path.parent.mkdir(parents=True)
    held = FileLock(lock_path)
    with held:
        with pytest.raises(ModelCliError) as raised:
            with model_lock("asr", tmp_path, timeout=0.01):
                pass

    assert raised.value.code == "MODEL_BUSY"
    assert raised.value.retryable is True


def test_model_lock_does_not_reclassify_body_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="inference read failed"):
        with model_lock("asr", tmp_path):
            raise OSError("inference read failed")


def test_refresh_publish_failure_restores_old_model(monkeypatch, isolated_cache: Path, tmp_path: Path) -> None:
    old_dir = _write_asr_cache(isolated_cache, b"old")
    _create_asr_manifest(isolated_cache, old_dir)
    source = tmp_path / "refresh"
    new_dir = _write_asr_cache(source, b"new")
    _create_asr_manifest(source, new_dir)
    monkeypatch.setattr(
        lifecycle,
        "verify_manifest",
        lambda *_args: (_ for _ in ()).throw(ModelCliError("BROKEN", "publish failed", 4)),
    )

    with pytest.raises(ModelCliError, match="publish failed"):
        lifecycle._publish_refresh("asr", source)

    assert (old_dir / "tokens.json").read_bytes() == b"old"
    manifest = json.loads(manifest_path(isolated_cache, "asr").read_text())
    assert manifest["files"]


def test_refresh_publish_replaces_model_and_manifest(isolated_cache: Path, tmp_path: Path) -> None:
    old_dir = _write_asr_cache(isolated_cache, b"old")
    _create_asr_manifest(isolated_cache, old_dir)
    source = tmp_path / "refresh-success"
    new_dir = _write_asr_cache(source, b"new")
    _create_asr_manifest(source, new_dir)

    lifecycle._publish_refresh("asr", source)

    assert (old_dir / "tokens.json").read_bytes() == b"new"
    assert verify_manifest(isolated_cache, "asr")["verified"] is True


def test_models_status_reports_manifest_metadata(isolated_cache: Path) -> None:
    model_dir = _write_asr_cache(isolated_cache)
    _create_asr_manifest(isolated_cache, model_dir)

    status = lifecycle.get_model_status("asr")

    assert status.status == "installed"
    assert status.manifest_status == "present"
    assert status.model == "SenseVoiceSmall INT8 ONNX"
    assert status.requested_revision == "v2.0.5"
    assert status.source_revision is None


def test_download_asr_uses_pinned_quantized_model_and_verified_tokenizer(
    monkeypatch,
    isolated_cache: Path,
) -> None:
    model_dir = lifecycle.asr_cache_dir(isolated_cache)
    calls: list[tuple] = []

    def ensure_modelscope(model_id: str, revision: str, *, cache_root: Path) -> Path:
        calls.append(("model", model_id, revision, cache_root))
        model_dir.mkdir(parents=True)
        for filename in lifecycle.ASR_DOWNLOAD_FILES:
            (model_dir / filename).write_bytes(b"model")
        (model_dir / ".download_ok").write_bytes(b"ok")
        return model_dir

    def ensure_file(
        url: str,
        filename: str,
        dest_dir: Path,
        *,
        expected_sha256: str,
    ) -> Path:
        calls.append(("tokenizer", url, filename, dest_dir, expected_sha256))
        path = dest_dir / filename
        path.write_bytes(b"tokenizer")
        return path

    monkeypatch.setattr("modelcli.models.cache.ensure_modelscope", ensure_modelscope)
    monkeypatch.setattr("modelcli.models.cache.ensure_file_from_url", ensure_file)

    assert lifecycle._download_asr(isolated_cache) == model_dir
    assert calls == [
        (
            "model",
            "iic/SenseVoiceSmall-onnx",
            "v2.0.5",
            isolated_cache,
        ),
        (
            "tokenizer",
            lifecycle.SENSEVOICE_TOKENIZER_URL,
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
            model_dir,
            lifecycle.SENSEVOICE_TOKENIZER_SHA256,
        ),
    ]


def test_load_asr_enables_quantized_onnx(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, int, bool]] = []

    class FakeSenseVoice:
        def __init__(self, path: str, *, batch_size: int, quantize: bool) -> None:
            calls.append((path, batch_size, quantize))

    monkeypatch.setattr("funasr_onnx.SenseVoiceSmall", FakeSenseVoice)

    lifecycle._load_asr(tmp_path)

    assert calls == [(str(tmp_path), 1, True)]


def test_stale_legacy_manifest_does_not_block_quantized_install(
    monkeypatch,
    isolated_cache: Path,
) -> None:
    legacy_dir = isolated_cache / "iic__SenseVoiceSmall"
    legacy_dir.mkdir()
    legacy_artifact = legacy_dir / "model.onnx"
    legacy_artifact.write_bytes(b"legacy")
    create_manifest(
        isolated_cache,
        "asr",
        ["iic/SenseVoiceSmall"],
        [legacy_artifact],
        requested_revision="master",
    )
    legacy_artifact.unlink()

    def download(root: Path) -> Path:
        return _write_asr_cache(root)

    monkeypatch.setattr(lifecycle, "_download_asr", download)
    monkeypatch.setattr(lifecycle, "_load_asr", lambda path: None)

    result = lifecycle.prepare_asr_model(validate=True, allow_download=True)

    model_dir = lifecycle.asr_cache_dir(isolated_cache)
    assert result == model_dir
    manifest = json.loads(manifest_path(isolated_cache, "asr").read_text())
    assert manifest["model_ids"] == ["iic/SenseVoiceSmall-onnx"]
    assert manifest["requested_revision"] == "v2.0.5"
    assert legacy_dir.exists()


def test_complete_quantized_cache_loads_before_replacing_stale_manifest(
    monkeypatch,
    isolated_cache: Path,
) -> None:
    legacy_artifact = isolated_cache / "iic__SenseVoiceSmall" / "model.onnx"
    legacy_artifact.parent.mkdir()
    legacy_artifact.write_bytes(b"legacy")
    create_manifest(
        isolated_cache,
        "asr",
        ["iic/SenseVoiceSmall"],
        [legacy_artifact],
        requested_revision="master",
    )
    model_dir = _write_asr_cache(isolated_cache)

    def load(path: Path) -> None:
        assert path == model_dir
        manifest = json.loads(manifest_path(isolated_cache, "asr").read_text())
        assert manifest["model_ids"] == ["iic/SenseVoiceSmall"]

    monkeypatch.setattr(lifecycle, "_load_asr", load)

    lifecycle.prepare_asr_model(validate=False)

    manifest = json.loads(manifest_path(isolated_cache, "asr").read_text())
    assert manifest["model_ids"] == ["iic/SenseVoiceSmall-onnx"]
    assert manifest["requested_revision"] == "v2.0.5"


def test_verify_loads_quantized_cache_before_replacing_stale_manifest(
    monkeypatch,
    isolated_cache: Path,
) -> None:
    legacy_artifact = isolated_cache / "iic__SenseVoiceSmall" / "model.onnx"
    legacy_artifact.parent.mkdir()
    legacy_artifact.write_bytes(b"legacy")
    create_manifest(
        isolated_cache,
        "asr",
        ["iic/SenseVoiceSmall"],
        [legacy_artifact],
        requested_revision="master",
    )
    model_dir = _write_asr_cache(isolated_cache)
    calls: list[Path] = []
    monkeypatch.setattr(lifecycle, "_load_asr", calls.append)

    result = lifecycle.verify_models(lifecycle.ModelTarget.asr)

    assert calls == [model_dir]
    assert result[0]["requested_revision"] == "v2.0.5"
