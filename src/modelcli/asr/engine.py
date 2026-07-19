"""ASR engine backed by SenseVoice-Small via funasr-onnx (ONNXRuntime)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from modelcli.config import ASR_SAMPLE_RATE

# SenseVoice special-token regex: <|NAME|>
_TOKEN_RE = re.compile(r"<\|[^|]+\|>")
_EMOTION_TOKENS = {
    "HAPPY",
    "SAD",
    "ANGRY",
    "NEUTRAL",
    "FEARFUL",
    "DISGUSTED",
    "SURPRISED",
    "EMO_UNKNOWN",
}
_EVENT_TOKENS = {"BGM", "Applause", "Laughter", "Cough", "Sneeze", "Cry", "Sigh", "Speech"}


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    emotion: str | None = None
    events: tuple[str, ...] = ()


@dataclass
class AsrResult:
    text: str
    segments: list[AsrSegment]

    def to_dict(self, *, language: str) -> dict:
        return {
            "text": self.text,
            "language": language,
            "segments": [asdict(segment) for segment in self.segments],
        }


class AsrEngine:
    """Quantized SenseVoice-Small ONNX ASR with optional VAD segmentation."""

    def __init__(self, lang: str = "auto") -> None:
        self.lang = lang
        self._model = None
        self._model_dir: Path | None = None

    def _ensure_model(self) -> Path:
        if self._model_dir is None:
            from modelcli.models.lifecycle import prepare_asr_model

            self._model_dir = prepare_asr_model(validate=False)
        return self._model_dir

    def transcribe(
        self,
        audio_path: Path,
        use_vad: bool = True,
        with_emotion: bool = False,
    ) -> AsrResult:
        model_dir = self._ensure_model()
        audio, sr = _load_audio(audio_path, ASR_SAMPLE_RATE)

        if use_vad:
            from modelcli.asr.vad import VadSegmenter

            vad = VadSegmenter()
            segments_bounds = vad.split(audio, sr=ASR_SAMPLE_RATE)
            if not segments_bounds:
                return AsrResult(text="", segments=[])
        else:
            segments_bounds = [(0.0, len(audio) / sr)]

        segments = self._transcribe_with_funasr_onnx(
            model_dir, audio, sr, segments_bounds, with_emotion
        )

        segments = [
            segment
            for segment in segments
            if segment.text.strip()
            or (with_emotion and (segment.emotion or segment.events))
        ]

        full_text = _join_segment_texts(s.text for s in segments)
        return AsrResult(text=full_text, segments=segments)

    # ---- funasr-onnx path ----
    def _transcribe_with_funasr_onnx(
        self,
        model_dir: Path,
        audio: np.ndarray,
        sr: int,
        segments_bounds: list[tuple[float, float]],
        with_emotion: bool,
    ) -> list[AsrSegment]:
        from funasr_onnx import SenseVoiceSmall

        model = SenseVoiceSmall(str(model_dir), batch_size=1, quantize=True)
        results: list[AsrSegment] = []
        for start, end in segments_bounds:
            s = int(start * sr)
            e = int(end * sr)
            chunk = audio[s:e]
            res = model(chunk, language=self.lang, use_itn=True)
            # funasr-onnx returns a list of strings, each already a transcript.
            if isinstance(res, list) and res:
                raw_text = str(res[0])
            else:
                raw_text = str(res)
            text, emotion, events = _clean_text(raw_text, with_emotion)
            results.append(
                AsrSegment(
                    start=start,
                    end=end,
                    text=text,
                    emotion=emotion,
                    events=events,
                )
            )
        return results


def _load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample_poly

        g = np.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return audio, sr


def _clean_text(
    raw: str,
    with_emotion: bool,
) -> tuple[str, str | None, tuple[str, ...]]:
    """Strip SenseVoice tokens and optionally return emotion and event metadata."""
    emotion: str | None = None
    events: list[str] = []

    def _handle(m: re.Match) -> str:
        nonlocal emotion
        name = m.group(0)[2:-2]
        if with_emotion and name in _EMOTION_TOKENS and name != "EMO_UNKNOWN":
            emotion = name.lower()
        if with_emotion and name in _EVENT_TOKENS and name not in events:
            events.append(name)
        return ""

    cleaned = _TOKEN_RE.sub(_handle, raw)
    text = re.sub(r"\s+", " ", cleaned).strip()
    return text, emotion, tuple(events)


def _join_segment_texts(texts: Iterable[str]) -> str:
    """Join ASR segments without gluing adjacent Latin words together."""
    result = ""
    for text in texts:
        if not text:
            continue
        if result and _needs_word_separator(result[-1], text[0]):
            result += " "
        result += text
    return result


def _needs_word_separator(left: str, right: str) -> bool:
    return left.isascii() and right.isascii() and left.isalnum() and right.isalnum()
