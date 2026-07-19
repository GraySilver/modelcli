from __future__ import annotations

import json

from typer.testing import CliRunner

from modelcli.cli import app
from modelcli.errors import model_error
from modelcli.models import lifecycle

runner = CliRunner()


def _payload(result) -> dict:
    return json.loads(result.stdout)


def test_capabilities_agent_contract() -> None:
    result = runner.invoke(app, ["--json", "capabilities"])

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["operation"] == "capabilities"
    assert payload["result"]["agent_policy"]["implicit_download"] is False
    assert payload["result"]["agent_policy"]["inference_timeout_owner"] == "caller"
    assert payload["result"]["detect"]["model"] == "PicoDet-L 416 COCO"
    assert payload["result"]["detect"]["provider"] == "CPUExecutionProvider"
    assert payload["result"]["asr"]["model"] == "SenseVoiceSmall INT8 ONNX"
    assert payload["result"]["tts"]["sample_rate"] == 48_000


def test_doctor_agent_contract_without_deep(monkeypatch) -> None:
    monkeypatch.setattr("modelcli.diagnostics.list_models", lambda: [])

    result = runner.invoke(app, ["--json", "doctor"])

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["operation"] == "doctor"
    assert payload["result"]["deep"] is False
    assert all("deep_load" not in check["name"] for check in payload["result"]["checks"])


def test_models_list_uses_global_agent_envelope(monkeypatch) -> None:
    monkeypatch.setattr(
        lifecycle,
        "list_models",
        lambda: [
            lifecycle.ModelStatus("asr", "SenseVoiceSmall INT8 ONNX", "missing", 0, True)
        ],
    )

    result = runner.invoke(app, ["--json", "models", "list"])

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["operation"] == "models.list"
    assert payload["result"]["models"][0]["manifest_status"] == "not_applicable"


def test_models_install_passes_refresh_and_returns_envelope(monkeypatch) -> None:
    calls: list[tuple[lifecycle.ModelTarget, bool]] = []

    def install(target: lifecycle.ModelTarget, *, refresh: bool):
        calls.append((target, refresh))
        return [
            lifecycle.ModelActionResult(
                "asr", "installed", True, 123, "present", "verified", "v2.0.5", None
            )
        ]

    monkeypatch.setattr(lifecycle, "install_models", install)
    result = runner.invoke(app, ["--json", "models", "install", "asr", "--refresh"])

    assert result.exit_code == 0
    assert calls == [(lifecycle.ModelTarget.asr, True)]
    payload = _payload(result)
    assert payload["operation"] == "models.install"
    assert payload["result"]["refresh"] is True
    assert payload["result"]["models"][0]["verification_status"] == "verified"


def test_models_verify_and_remove_operations(monkeypatch) -> None:
    monkeypatch.setattr(
        lifecycle,
        "verify_models",
        lambda target: [{"name": target.value, "verified": True, "files_checked": 3}],
    )
    monkeypatch.setattr(
        lifecycle,
        "remove_models",
        lambda target: [lifecycle.ModelActionResult(target.value, "missing", True, 0)],
    )

    verified = runner.invoke(app, ["--json", "models", "verify", "asr"])
    removed = runner.invoke(app, ["--json", "models", "remove", "tts"])

    assert verified.exit_code == 0
    assert _payload(verified)["operation"] == "models.verify"
    assert removed.exit_code == 0
    assert _payload(removed)["operation"] == "models.remove"


def test_model_errors_use_exit_code_four(monkeypatch) -> None:
    monkeypatch.setattr(
        lifecycle,
        "install_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            model_error("MODEL_DOWNLOAD_FAILED", "network unavailable", retryable=True)
        ),
    )

    result = runner.invoke(app, ["--json", "models", "install", "asr"])

    assert result.exit_code == 4
    assert _payload(result)["error"] == {
        "code": "MODEL_DOWNLOAD_FAILED",
        "message": "network unavailable",
        "retryable": True,
    }
