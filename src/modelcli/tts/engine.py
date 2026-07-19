"""TTS engine backed by MOSS-TTS-Nano (0.1B, 48 kHz stereo, voice-clone mode).

Models come from ModelScope (`OpenMOSS/MOSS-TTS-Nano` main model +
`openmoss/MOSS-Audio-Tokenizer-Nano`). Default voice is the official Chinese
female reference audio `zh_1.wav`, downloaded into the cache on first use;
callers can pass their own reference audio via `prompt_audio`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM

from modelcli.config import (
    CACHE_ROOT,
    MOSS_DEFAULT_PROMPT_NAME,
    MOSS_DEFAULT_PROMPT_URL,
    MODELSCOPE_MOSS_AUDIO_TOKENIZER,
    MODELSCOPE_MOSS_TTS,
    TTS_SAMPLE_RATE,
)
from modelcli.models.cache import ensure_file_from_url, ensure_modelscope


@dataclass
class TtsResult:
    audio: np.ndarray  # float32 (N, 2) stereo at 48 kHz
    sample_rate: int


_PROMPTS_DIR = CACHE_ROOT / "moss_prompts"


def default_prompt_audio() -> Path:
    """Return path to the bundled Chinese reference prompt, downloading if needed."""
    _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    return ensure_file_from_url(MOSS_DEFAULT_PROMPT_URL, MOSS_DEFAULT_PROMPT_NAME, _PROMPTS_DIR)


class TtsEngine:
    """MOSS-TTS-Nano wrapper. Lazy-loads the model on first synthesize()."""

    def __init__(self) -> None:
        self._model = None
        self._main_dir: Path | None = None
        self._tok_dir: Path | None = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model, self._main_dir, self._tok_dir

        self._main_dir = ensure_modelscope(MODELSCOPE_MOSS_TTS)
        self._tok_dir = ensure_modelscope(MODELSCOPE_MOSS_AUDIO_TOKENIZER)

        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        model = AutoModelForCausalLM.from_pretrained(
            str(self._main_dir),
            trust_remote_code=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device=device, dtype=torch.float32)
        model._set_attention_implementation("sdpa")
        model.eval()
        self._model = model
        return self._model, self._main_dir, self._tok_dir

    def synthesize(
        self,
        text: str,
        prompt_audio: Path | None = None,
        max_new_frames: int = 600,
    ) -> TtsResult:
        """Synthesize text in voice_clone mode using a reference audio.

        Args:
            text: Text to speak.
            prompt_audio: Reference audio for voice cloning. Defaults to the
                bundled Chinese female sample.
            max_new_frames: Upper bound on generated audio frames
                (1 frame = 80 ms at 12.5 Hz token rate).
        """
        model, main_dir, tok_dir = self._ensure_model()
        device = next(model.parameters()).device

        prompt_path = str(prompt_audio) if prompt_audio else str(default_prompt_audio())
        # Write to a temp wav first; model.inference reads audio from file path.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        try:
            model.inference(
                text=text,
                output_audio_path=out_path,
                mode="voice_clone",
                prompt_audio_path=prompt_path,
                audio_tokenizer_type="moss-audio-tokenizer-nano",
                audio_tokenizer_pretrained_name_or_path=str(tok_dir),
                device=device,
                max_new_frames=max_new_frames,
                do_sample=True,
                use_kv_cache=True,
            )
            audio, sr = sf.read(out_path, dtype="float32")
        finally:
            Path(out_path).unlink(missing_ok=True)

        return TtsResult(audio=np.asarray(audio, dtype=np.float32), sample_rate=int(sr))

    def synthesize_to_file(
        self,
        text: str,
        out: Path,
        prompt_audio: Path | None = None,
        max_new_frames: int = 600,
    ) -> TtsResult:
        result = self.synthesize(text, prompt_audio=prompt_audio, max_new_frames=max_new_frames)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), result.audio, result.sample_rate)
        return result


def play_result(result: TtsResult) -> None:
    """Play audio via simpleaudio. Raises ImportError with a friendly hint."""
    try:
        import simpleaudio as sa
    except ImportError as e:
        raise ImportError(
            "simpleaudio is required for --play. Install with: "
            "uv pip install '.[play]'  (or pip install simpleaudio)"
        ) from e

    # Mix stereo -> mono for playback (simpleaudio supports mono well)
    audio = np.asarray(result.audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    play_obj = sa.play_buffer(pcm_int16, 1, 2, result.sample_rate)
    play_obj.wait_done()
