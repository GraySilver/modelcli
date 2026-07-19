from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from typer.testing import CliRunner

from modelcli.asr.engine import AsrResult, AsrSegment
from modelcli.cli import app

runner = CliRunner()


def test_ocr_stdout_is_content_only_and_preserves_markup(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"image")

    class FakeResult:
        def to_text(self) -> str:
            return "literal [red]error[/red]"

        def to_json(self) -> str:
            return '{"text":"[red]error[/red]"}'

        def to_markdown(self) -> str:
            return "[heading]"

    class FakeEngine:
        def recognize(self, _image: Path) -> FakeResult:
            return FakeResult()

    monkeypatch.setattr("modelcli.ocr.engine.OcrEngine", FakeEngine)

    result = runner.invoke(app, ["ocr", str(image), "--json"])

    assert result.exit_code == 0
    assert result.stdout == '{"text":"[red]error[/red]"}\n'
    assert "Loading OCR model" not in result.stdout


def test_ocr_out_writes_file_without_stdout(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    output = tmp_path / "result.txt"

    class FakeResult:
        def to_text(self) -> str:
            return "recognized"

        to_json = to_text
        to_markdown = to_text

    class FakeEngine:
        def recognize(self, _image: Path) -> FakeResult:
            return FakeResult()

    monkeypatch.setattr("modelcli.ocr.engine.OcrEngine", FakeEngine)

    result = runner.invoke(app, ["ocr", str(image), "--out", str(output)])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert output.read_text(encoding="utf-8") == "recognized"
    assert "Wrote result" in result.stderr


def test_asr_emotion_and_timestamps_have_stable_format(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")

    class FakeEngine:
        def __init__(self, lang: str) -> None:
            assert lang == "en"

        def transcribe(self, *_args, **_kwargs) -> AsrResult:
            return AsrResult(
                text="hello world",
                segments=[
                    AsrSegment(
                        start=0.0,
                        end=1.25,
                        text="hello world",
                        emotion="neutral",
                        events=("Speech", "Applause"),
                    )
                ],
            )

    monkeypatch.setattr("modelcli.asr.engine.AsrEngine", FakeEngine)

    result = runner.invoke(
        app,
        ["asr", str(audio), "--lang", "en", "--timestamps", "--emotion"],
    )

    assert result.exit_code == 0
    assert result.stdout == (
        "[0.00-1.25] hello world "
        "(emotion=neutral, events=Speech|Applause)\n"
    )


def test_asr_emotion_without_timestamps_is_segmented(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")

    class FakeEngine:
        def __init__(self, lang: str) -> None:
            pass

        def transcribe(self, *_args, **_kwargs) -> AsrResult:
            return AsrResult(
                text="first second",
                segments=[
                    AsrSegment(0.0, 1.0, "first", emotion="happy"),
                    AsrSegment(1.0, 2.0, "second", events=("Laughter",)),
                ],
            )

    monkeypatch.setattr("modelcli.asr.engine.AsrEngine", FakeEngine)

    result = runner.invoke(app, ["asr", str(audio), "--emotion"])

    assert result.exit_code == 0
    assert result.stdout == (
        "first (emotion=happy)\n"
        "second (events=Laughter)\n"
    )


def test_invalid_language_is_usage_error(tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")

    invalid_language = runner.invoke(app, ["asr", str(audio), "--lang", "invalid"])

    assert invalid_language.exit_code == 2
    assert "Traceback" not in invalid_language.output


def test_tts_max_duration_zero_is_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["tts", "hello", "--max-duration", "0"])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_tts_status_does_not_pollute_stdout(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "speech.wav"

    class FakeEngine:
        def synthesize_to_file(self, _text: str, out: Path, **_kwargs):
            out.write_bytes(b"wave")
            return SimpleNamespace(audio=np.zeros(24_000), sample_rate=24_000)

    monkeypatch.setattr("modelcli.tts.engine.TtsEngine", FakeEngine)

    result = runner.invoke(app, ["tts", "hello", "--out", str(output)])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "Saved" in result.stderr
