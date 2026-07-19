from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from modelcli.asr.engine import AsrResult, AsrSegment
from modelcli.cli import app
from modelcli.ocr.engine import OcrLine, OcrResult

runner = CliRunner()


def _json(result) -> dict:
    return json.loads(result.stdout)


def _image(path: Path) -> None:
    cv2.imwrite(str(path), np.full((16, 24, 3), 255, dtype=np.uint8))


def _audio(path: Path) -> None:
    sf.write(str(path), np.zeros(800, dtype=np.float32), 16_000)


def test_version_supports_human_and_agent_modes() -> None:
    human = runner.invoke(app, ["--version"])
    agent = runner.invoke(app, ["--json", "--version"])

    assert human.exit_code == 0
    assert human.stdout == "0.2.0\n"
    assert _json(agent)["result"] == {"version": "0.2.0"}


def test_agent_ocr_returns_single_structured_envelope(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    _image(image)

    class FakeEngine:
        def recognize(self, _image: Path) -> OcrResult:
            return OcrResult([OcrLine("hello", 0.98, [[0, 0], [1, 0], [1, 1], [0, 1]])])

    monkeypatch.setattr("modelcli.ocr.engine.OcrEngine", FakeEngine)

    result = runner.invoke(app, ["--json", "ocr", str(image)])

    assert result.exit_code == 0
    assert result.stdout.count("\n") == 1
    payload = _json(result)
    assert payload["schema_version"] == "1"
    assert payload["ok"] is True
    assert payload["operation"] == "ocr"
    assert payload["result"]["lines"][0]["text"] == "hello"


def test_corrupt_image_is_compact_agent_error(tmp_path: Path) -> None:
    image = tmp_path / "bad.png"
    image.write_bytes(b"not-an-image")

    result = runner.invoke(app, ["--json", "ocr", str(image)])

    assert result.exit_code == 3
    assert result.stderr == ""
    assert "Traceback" not in result.output
    assert _json(result)["error"]["code"] == "INVALID_IMAGE"


def test_debug_prints_traceback_only_to_stderr(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    _image(image)

    class BrokenEngine:
        def recognize(self, _image: Path) -> OcrResult:
            raise RuntimeError("broken OCR")

    monkeypatch.setattr("modelcli.ocr.engine.OcrEngine", BrokenEngine)
    result = runner.invoke(app, ["--json", "--debug", "ocr", str(image)])

    assert result.exit_code == 5
    assert _json(result)["error"]["code"] == "INTERNAL_ERROR"
    assert "Traceback" in result.stderr
    assert "Traceback" not in result.stdout


def test_dependency_stdout_is_redirected_away_from_agent_envelope(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    _image(image)

    class NoisyEngine:
        def recognize(self, _image: Path) -> OcrResult:
            print("dependency noise")
            return OcrResult([])

    monkeypatch.setattr("modelcli.ocr.engine.OcrEngine", NoisyEngine)
    result = runner.invoke(app, ["--json", "ocr", str(image)])

    assert result.exit_code == 0
    assert result.stdout.count("\n") == 1
    assert _json(result)["ok"] is True
    assert "dependency noise" in result.stderr


def test_old_subcommand_json_option_is_rejected() -> None:
    result = runner.invoke(app, ["models", "list", "--json"])

    assert result.exit_code == 2
    assert "No such option: --json" in result.stderr


def test_agent_mode_without_command_is_usage_error() -> None:
    result = runner.invoke(app, ["--json"])

    assert result.exit_code == 2
    assert _json(result)["error"]["code"] == "CLI_USAGE_ERROR"


def test_agent_asr_result_contains_requested_language_and_segments(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    _audio(audio)

    class FakeEngine:
        def __init__(self, lang: str) -> None:
            assert lang == "en"

        def transcribe(self, *_args, **_kwargs) -> AsrResult:
            return AsrResult(
                "hello",
                [AsrSegment(0.0, 1.25, "hello", "neutral", ("Speech",))],
            )

    monkeypatch.setattr("modelcli.asr.engine.AsrEngine", FakeEngine)
    result = runner.invoke(app, ["--json", "asr", str(audio), "--lang", "en"])

    assert result.exit_code == 0
    payload = _json(result)["result"]
    assert payload["text"] == "hello"
    assert payload["language"] == "en"
    assert payload["segments"] == [
        {"start": 0.0, "end": 1.25, "text": "hello", "emotion": "neutral", "events": ["Speech"]}
    ]


def test_allow_download_is_visible_to_inference(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    _audio(audio)

    class FakeEngine:
        def __init__(self, lang: str) -> None:
            from modelcli.protocol import current_runtime

            assert current_runtime().allow_download is True

        def transcribe(self, *_args, **_kwargs) -> AsrResult:
            return AsrResult("", [])

    monkeypatch.setattr("modelcli.asr.engine.AsrEngine", FakeEngine)
    result = runner.invoke(app, ["--json", "--allow-download", "asr", str(audio)])

    assert result.exit_code == 0


def test_agent_tts_requires_explicit_output() -> None:
    result = runner.invoke(app, ["--json", "tts", "hello"])

    assert result.exit_code == 3
    assert _json(result)["error"]["code"] == "OUTPUT_REQUIRED"


def test_tts_rejects_existing_output_before_loading_engine(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "speech.wav"
    output.write_bytes(b"old")

    class UnexpectedEngine:
        def __init__(self) -> None:
            raise AssertionError("engine must not load")

    monkeypatch.setattr("modelcli.tts.engine.TtsEngine", UnexpectedEngine)
    result = runner.invoke(app, ["--json", "tts", "hello", "--out", str(output)])

    assert result.exit_code == 6
    assert output.read_bytes() == b"old"
    assert _json(result)["error"]["code"] == "OUTPUT_EXISTS"


def test_ocr_rejects_existing_annotation_before_loading_engine(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    annotation = tmp_path / "annotated.png"
    _image(image)
    annotation.write_bytes(b"old")

    class UnexpectedEngine:
        def __init__(self) -> None:
            raise AssertionError("engine must not load")

    monkeypatch.setattr("modelcli.ocr.engine.OcrEngine", UnexpectedEngine)
    result = runner.invoke(
        app,
        ["--json", "ocr", str(image), "--draw-boxes", str(annotation)],
    )

    assert result.exit_code == 6
    assert annotation.read_bytes() == b"old"
    assert _json(result)["error"]["code"] == "OUTPUT_EXISTS"


def test_agent_tts_returns_file_metadata(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "speech.wav"

    class FakeEngine:
        def synthesize_to_file(self, _text: str, out: Path, **_kwargs):
            sf.write(str(out), np.zeros((480, 2), dtype=np.float32), 48_000)
            return SimpleNamespace(
                audio=np.zeros((480, 2), dtype=np.float32),
                sample_rate=48_000,
                reached_frame_cap=False,
            )

    monkeypatch.setattr("modelcli.tts.engine.TtsEngine", FakeEngine)
    result = runner.invoke(app, ["--json", "tts", "hello", "--out", str(output)])

    assert result.exit_code == 0
    payload = _json(result)["result"]
    assert payload["output"] == str(output.resolve())
    assert payload["size_bytes"] == output.stat().st_size
    assert payload["sample_rate"] == 48_000
    assert payload["channels"] == 2
    assert payload["prompt_source"]["type"] == "default"
    assert payload["reached_frame_cap"] is False


def test_human_asr_format_remains_plain_text(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    _audio(audio)

    class FakeEngine:
        def __init__(self, lang: str) -> None:
            pass

        def transcribe(self, *_args, **_kwargs) -> AsrResult:
            return AsrResult("hello", [AsrSegment(0.0, 1.0, "hello", "neutral", ("Speech",))])

    monkeypatch.setattr("modelcli.asr.engine.AsrEngine", FakeEngine)
    result = runner.invoke(app, ["asr", str(audio), "--timestamps", "--emotion"])

    assert result.exit_code == 0
    assert result.stdout == "[0.00-1.00] hello (emotion=neutral, events=Speech)\n"
