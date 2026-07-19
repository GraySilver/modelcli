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

@dataclass
class TtsResult:
    audio: np.ndarray  # float32 (N, 2) stereo at 48 kHz
    sample_rate: int
    reached_frame_cap: bool = False


def default_prompt_audio() -> Path:
    """Return the installed default Chinese reference prompt."""
    from modelcli.config import CACHE_ROOT
    from modelcli.models.locking import model_lock
    from modelcli.models.lifecycle import prepare_tts_model

    with model_lock("tts", CACHE_ROOT):
        return prepare_tts_model(validate=False)[2]


class TtsEngine:
    """MOSS-TTS-Nano wrapper. Lazy-loads the model on first synthesize()."""

    def __init__(self) -> None:
        self._model = None
        self._main_dir: Path | None = None
        self._tok_dir: Path | None = None
        self._prompt_path: Path | None = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model, self._main_dir, self._tok_dir

        from modelcli.models.lifecycle import prepare_tts_model

        self._main_dir, self._tok_dir, self._prompt_path = prepare_tts_model(validate=False)

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
        model, _main_dir, tok_dir = self._ensure_model()
        device = next(model.parameters()).device

        prompt_path = str(prompt_audio) if prompt_audio else str(self._prompt_path)
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

        audio_array = np.asarray(audio, dtype=np.float32)
        frame_count = len(audio_array) / (int(sr) * 0.08)
        return TtsResult(
            audio=audio_array,
            sample_rate=int(sr),
            reached_frame_cap=frame_count >= max_new_frames - 0.5,
        )

    def synthesize_to_file(
        self,
        text: str,
        out: Path,
        prompt_audio: Path | None = None,
        max_new_frames: int = 600,
        force: bool = False,
    ) -> TtsResult:
        from modelcli.errors import output_error
        from modelcli.files import atomic_output_path

        with atomic_output_path(out, force=force) as temporary:
            result = self.synthesize(
                text,
                prompt_audio=prompt_audio,
                max_new_frames=max_new_frames,
            )
            try:
                sf.write(str(temporary), result.audio, result.sample_rate)
                with sf.SoundFile(str(temporary)) as output_file:
                    if output_file.frames <= 0:
                        raise RuntimeError("empty audio")
            except Exception as exc:
                raise output_error("OUTPUT_WRITE_FAILED", f"Cannot write audio output: {out}") from exc
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
