"""Voice Activity Detection via the official silero-vad pip package.

silero-vad ships its own ONNX-exported torchscript model and exposes
`load_silero_vad()` + `get_speech_timestamps()`. We read audio ourselves with
soundfile (to avoid a torchaudio/torchcodec dependency on macOS) and feed it
in as a 16 kHz mono float32 torch tensor.
"""

from __future__ import annotations

import numpy as np
import torch

SR = 16000
THRESHOLD = 0.5
MIN_SPEECH_MS = 250
MIN_SILENCE_MS = 500
SPEECH_PAD_MS = 30


class VadSegmenter:
    """Wrapper around silero-vad. Input: mono float32 numpy audio at any SR."""

    def __init__(self, threshold: float = THRESHOLD) -> None:
        self.threshold = threshold
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
        return self._model

    def split(
        self,
        audio: np.ndarray,
        sr: int = SR,
    ) -> list[tuple[float, float]]:
        """Return list of (start_sec, end_sec) speech segments."""
        if sr != SR:
            from scipy.signal import resample_poly

            g = int(np.gcd(sr, SR))
            audio = resample_poly(audio, SR // g, sr // g).astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        wav = torch.from_numpy(audio)
        model = self._ensure_model()
        from silero_vad import get_speech_timestamps

        ts = get_speech_timestamps(
            wav,
            model,
            sampling_rate=SR,
            threshold=self.threshold,
            min_speech_duration_ms=MIN_SPEECH_MS,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
            return_seconds=False,
        )
        return [(t["start"] / SR, t["end"] / SR) for t in ts]
