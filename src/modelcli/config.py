"""Global configuration constants for modelcli."""

from pathlib import Path

from platformdirs import user_cache_dir

# --- Model cache directory ---
CACHE_ROOT = Path(user_cache_dir("modelcli"))
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# --- ModelScope model IDs ---
MODELSCOPE_SENSEVOICE = "iic/SenseVoiceSmall-onnx"
MODELSCOPE_SENSEVOICE_REVISION = "v2.0.5"

# funasr-onnx requires this tokenizer, but the official ONNX repository does
# not ship it. Fetch the matching file from the official source repository.
SENSEVOICE_TOKENIZER_NAME = "chn_jpn_yue_eng_ko_spectok.bpe.model"
SENSEVOICE_TOKENIZER_URL = (
    "https://modelscope.cn/api/v1/models/iic/SenseVoiceSmall/repo"
    "?Revision=master&FilePath=chn_jpn_yue_eng_ko_spectok.bpe.model"
)
SENSEVOICE_TOKENIZER_SHA256 = (
    "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8"
)

# MOSS-TTS-Nano
MODELSCOPE_MOSS_TTS = "OpenMOSS/MOSS-TTS-Nano"
MODELSCOPE_MOSS_AUDIO_TOKENIZER = "openmoss/MOSS-Audio-Tokenizer-Nano"

# Default Chinese prompt audio (female reference shipped with MOSS-TTS-Nano repo).
# Downloaded once into the cache on first use.
MOSS_DEFAULT_PROMPT_URL = (
    "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS-Nano/"
    "11619374849c649486584e3b10ed55b176a924ee/assets/audio/zh_1.wav"
)
MOSS_DEFAULT_PROMPT_NAME = "zh_1.wav"
MOSS_DEFAULT_PROMPT_SHA256 = (
    "f64a53490acf7337358832d1da7c562e08e2f3caf4385727b318cdc4f4da50d2"
)

MODELSCOPE_REVISION = "master"
DOWNLOAD_TIMEOUT_SECONDS = 120
MODEL_LOCK_TIMEOUT_SECONDS = 30.0

# --- Object detection (PicoDet-L 416 COCO, postprocessed ONNX) ---
PICODET_MODEL_ID = "PaddleDetection/PicoDet-L-416-COCO"
PICODET_MODEL_REVISION = "release/2.8"
PICODET_MODEL_URL = (
    "https://paddledet.bj.bcebos.com/deploy/third_engine/"
    "picodet_l_416_lcnet_postprocessed.onnx"
)
PICODET_MODEL_NAME = "picodet_l_416_lcnet_postprocessed.onnx"
PICODET_MODEL_SHA256 = (
    "c5550352511bd46ff271711278a34e302d451cf172a1ea65e14cdc7da2a0a112"
)
PICODET_MODEL_SIZE = 23_244_019
PICODET_CACHE_NAME = "picodet_l_416_coco"
PICODET_INPUT_SIZE = 416
DETECT_DEFAULT_CONFIDENCE = 0.5

# --- ASR defaults ---
ASR_SAMPLE_RATE = 16000
ASR_DEFAULT_LANG = "auto"  # auto | zh | en | yue | ja | ko

# --- TTS defaults (MOSS-TTS-Nano) ---
# Output is fixed at 48 kHz stereo by the model.
TTS_SAMPLE_RATE = 48000

# --- ONNX Runtime providers (auto-detected at runtime) ---
# We let onnxruntime pick the best available provider.
