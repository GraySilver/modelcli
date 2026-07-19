from __future__ import annotations

from pathlib import Path

import numpy as np

from modelcli.asr.engine import AsrEngine, AsrSegment, _clean_text, _join_segment_texts


def test_clean_text_preserves_english_spaces_and_extracts_metadata() -> None:
    raw = (
        "<|en|><|NEUTRAL|><|Speech|><|woitn|>hello world"
        "<|Applause|><|Speech|>"
    )

    text, emotion, events = _clean_text(raw, with_emotion=True)

    assert text == "hello world"
    assert emotion == "neutral"
    assert events == ("Speech", "Applause")


def test_clean_text_discards_all_metadata_when_not_requested() -> None:
    raw = "<|zh|><|HAPPY|><|Laughter|><|woitn|>你好 世界"

    text, emotion, events = _clean_text(raw, with_emotion=False)

    assert text == "你好 世界"
    assert emotion is None
    assert events == ()


def test_clean_text_ignores_unknown_emotion() -> None:
    text, emotion, events = _clean_text(
        "<|en|><|EMO_UNKNOWN|><|BGM|>background music",
        with_emotion=True,
    )

    assert text == "background music"
    assert emotion is None
    assert events == ("BGM",)


def test_join_segments_separates_latin_words_without_changing_cjk() -> None:
    assert _join_segment_texts(["hello", "world 123"]) == "hello world 123"
    assert _join_segment_texts(["你好", "世界"]) == "你好世界"
    assert _join_segment_texts(["版本", "2 ready"]) == "版本2 ready"


def test_transcribe_keeps_event_only_segment_when_metadata_requested(
    monkeypatch,
) -> None:
    engine = AsrEngine()
    monkeypatch.setattr(engine, "_ensure_model", lambda: Path("/model"))
    monkeypatch.setattr(
        "modelcli.asr.engine._load_audio",
        lambda _path, _sample_rate: (np.zeros(16_000, dtype=np.float32), 16_000),
    )
    monkeypatch.setattr(
        engine,
        "_transcribe_with_funasr_onnx",
        lambda *_args: [AsrSegment(0.0, 1.0, "", events=("Applause",))],
    )

    result = engine.transcribe(
        Path("audio.wav"),
        use_vad=False,
        with_emotion=True,
    )

    assert result.text == ""
    assert result.segments == [
        AsrSegment(0.0, 1.0, "", events=("Applause",)),
    ]


def test_asr_engine_uses_model_lifecycle_prepare(monkeypatch, tmp_path: Path) -> None:
    calls: list[bool] = []

    def prepare(*, validate: bool) -> Path:
        calls.append(validate)
        return tmp_path

    monkeypatch.setattr("modelcli.models.lifecycle.prepare_asr_model", prepare)

    engine = AsrEngine()

    assert engine._ensure_model() == tmp_path
    assert engine._ensure_model() == tmp_path
    assert calls == [False]


def test_asr_inference_uses_quantized_onnx(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, int, bool]] = []

    class FakeSenseVoice:
        def __init__(self, path: str, *, batch_size: int, quantize: bool) -> None:
            calls.append((path, batch_size, quantize))

        def __call__(self, audio, *, language: str, use_itn: bool):
            assert audio.shape == (16_000,)
            assert language == "en"
            assert use_itn is True
            return ["<|en|><|NEUTRAL|><|Speech|>hello"]

    monkeypatch.setattr("funasr_onnx.SenseVoiceSmall", FakeSenseVoice)
    engine = AsrEngine(lang="en")

    segments = engine._transcribe_with_funasr_onnx(
        tmp_path,
        np.zeros(16_000, dtype=np.float32),
        16_000,
        [(0.0, 1.0)],
        True,
    )

    assert calls == [(str(tmp_path), 1, True)]
    assert segments == [AsrSegment(0.0, 1.0, "hello", "neutral", ("Speech",))]
