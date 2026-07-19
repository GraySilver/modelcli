"""Global configuration constants for modelcli."""

from pathlib import Path

from platformdirs import user_cache_dir

# --- Model cache directory ---
CACHE_ROOT = Path(user_cache_dir("modelcli"))
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# --- ModelScope model IDs ---
MODELSCOPE_SENSEVOICE = "iic/SenseVoiceSmall"

# MOSS-TTS-Nano
MODELSCOPE_MOSS_TTS = "OpenMOSS/MOSS-TTS-Nano"
MODELSCOPE_MOSS_AUDIO_TOKENIZER = "openmoss/MOSS-Audio-Tokenizer-Nano"

# Default Chinese prompt audio (female reference shipped with MOSS-TTS-Nano repo).
# Downloaded once into the cache on first use.
MOSS_DEFAULT_PROMPT_URL = (
    "https://github.com/OpenMOSS/MOSS-TTS-Nano/raw/main/assets/audio/zh_1.wav"
)
MOSS_DEFAULT_PROMPT_NAME = "zh_1.wav"

# --- ASR defaults ---
ASR_SAMPLE_RATE = 16000
ASR_DEFAULT_LANG = "auto"  # auto | zh | en | yue | ja | ko

# --- TTS defaults (MOSS-TTS-Nano) ---
# Output is fixed at 48 kHz stereo by the model.
TTS_SAMPLE_RATE = 48000

# --- ONNX Runtime providers (auto-detected at runtime) ---
# We let onnxruntime pick the best available provider.
