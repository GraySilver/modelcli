# ModelCLI

[![Commit Activity](https://img.shields.io/github/commit-activity/m/GraySilver/modelcli)](https://github.com/GraySilver/modelcli/graphs/commit-activity)
[![Version](https://img.shields.io/badge/version-0.3.0-0A7B83)](https://github.com/GraySilver/modelcli/blob/main/src/modelcli/__init__.py)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://github.com/GraySilver/modelcli/blob/main/pyproject.toml)
[![Agent JSON](https://img.shields.io/badge/Agent%20JSON-v1-2F855A)](#agent-json-protocol)
[![License](https://img.shields.io/github/license/GraySilver/modelcli)](https://github.com/GraySilver/modelcli/blob/main/LICENSE)

[简体中文](./README.md) | **English**

[Agent Skill](#agent-skill) | [Quick Start](#quick-start) | [Showcase](#model-showcase) | [Examples](#command-examples) | [Agent JSON Protocol](#agent-json-protocol) | [Development](#development)

> Run object detection, OCR, speech recognition, and speech synthesis locally through one command-line interface for both humans and agents.

## Agent Skill

Send this one sentence directly to Codex, Claude Code, or OpenClaw:

```text
Install the ModelCLI Agent Skill from https://github.com/GraySilver/modelcli. Read install-skill.sh, choose the matching --target for Codex, Claude Code, or OpenClaw, install it, verify that SKILL.md exists in the correct user-level Skills directory, and tell me whether a restart is required.
```

The repository includes one unified [`modelcli` Agent Skill](./skills/modelcli/SKILL.md) for object detection, OCR, speech recognition, speech synthesis, model management, and diagnostics. Codex, Claude Code, and OpenClaw discover it automatically in this repository: Codex and OpenClaw use [`.agents/skills/modelcli`](./.agents/skills/modelcli), while Claude Code uses [`.claude/skills/modelcli`](./.claude/skills/modelcli).

After opening the repository, invoke it directly from your agent:

```text
# Codex
$modelcli extract the text from samples/images/ocr-sign.png

# Claude Code / OpenClaw
/modelcli detect people and cars in samples/images/detect-street.jpg
```

You can also install the skill manually in your personal directories to use it from other projects. The macOS and Linux installer targets all supported agents by default:

```bash
curl -LsSf https://raw.githubusercontent.com/GraySilver/modelcli/main/install-skill.sh | sh
```

Install for one agent or install from a local checkout:

```bash
curl -LsSf https://raw.githubusercontent.com/GraySilver/modelcli/main/install-skill.sh \
  | sh -s -- --target claude

./install-skill.sh --target codex
./install-skill.sh --target openclaw
./install-skill.sh --target all
```

Codex and OpenClaw share `~/.agents/skills/modelcli`; Claude Code uses `~/.claude/skills/modelcli`. Restart the selected agent after installation. To remove only copies managed by this installer:

```bash
./install-skill.sh --target all --uninstall
```

The skill invokes ModelCLI through a deterministic JSON wrapper. It runs the official `install.sh` when the `modelcli` command is missing and permits missing models to download during normal inference. Destructive or expensive operations, including overwriting output, refreshing or deleting model caches, and `doctor --deep`, still require explicit user confirmation.

ModelCLI packages several open-source small models behind a consistent CLI. Its default mode is designed for direct terminal use. With the global `--json` option, it emits a stable, machine-readable JSON envelope for JarvisBot, automation scripts, and other agents invoking it as a subprocess.

Once downloaded, models can run offline. The default cache directory is `~/Library/Caches/modelcli` on macOS. Other platforms follow the user cache directory conventions provided by [`platformdirs`](https://platformdirs.readthedocs.io/).

| Capability | Default model | Input | Output |
| --- | --- | --- | --- |
| Object detection | PicoDet-L 416 COCO | Image | 80 object classes, confidence scores, pixel coordinates, annotated images |
| OCR | PP-OCRv4 mobile | Image | Chinese/English text, lines, coordinates, confidence scores, annotated images |
| ASR | SenseVoiceSmall INT8 ONNX | WAV / FLAC / MP3 | Chinese, English, Cantonese, Japanese, or Korean text; segments, emotions, and events |
| TTS | MOSS-TTS-Nano | Text and optional reference audio | 48 kHz stereo cloned speech |

## Quick Start

Install with one command on macOS or Linux:

```bash
curl -LsSf https://raw.githubusercontent.com/GraySilver/modelcli/main/install.sh | sh
```

The installer installs `uv` when needed, creates an isolated environment with Python 3.12 and the repository's `uv.lock`, and exposes `modelcli` in `~/.local/bin`. Installing the application and its Python dependencies does not download inference models.

```bash
modelcli --version
modelcli doctor
modelcli models list
```

If your shell cannot find `modelcli`, add the user command directory to `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

To inspect the installer before running it:

```bash
curl -LsSf https://raw.githubusercontent.com/GraySilver/modelcli/main/install.sh -o install.sh
less install.sh
sh install.sh
```

To install with Python 3.11:

```bash
MODELCLI_PYTHON=3.11 sh install.sh
```

In human mode, the model required by an inference command is downloaded automatically the first time it is needed. To prepare all models in advance:

```bash
modelcli models prefetch
modelcli models verify all
```

The downloadable object detection, ASR, and TTS models total approximately 590 MB. OCR and VAD are bundled with the Python dependencies.

## Model Showcase

The following results were produced by the current version of ModelCLI. See [`samples/`](./samples/README_EN.md) for original inputs, complete outputs, reproduction commands, and asset licenses.

### Object detection

| Input | ModelCLI output |
| :---: | :---: |
| <img src="samples/images/detect-street.jpg" alt="Original Amsterdam street scene" width="360"> | <img src="samples/results/detect-street-boxes.jpg" alt="PicoDet-L detecting cars and people in an Amsterdam street scene" width="360"> |

At a `0.6` confidence threshold, PicoDet-L 416 COCO detects five people and two cars.

```bash
modelcli detect samples/images/detect-street.jpg \
  --confidence 0.6 \
  --draw-boxes detected.jpg
```

### OCR

<img src="samples/results/ocr-sign-boxes.png" alt="PP-OCRv4 Chinese and English recognition result" width="720">

```text
你好世界，这是一个OCR测试
HelloWorld 12345
```

### ASR and TTS

- [Listen to the ASR input](./samples/audio/asr-zh.wav) -> [view the actual SenseVoiceSmall transcript](./samples/results/asr-zh.txt)
- [Listen to the MOSS-TTS-Nano output](./samples/audio/tts-zh.wav)

```text
[0.10-0.54] 你好
[1.19-2.43] 欢迎使用本地模型
[3.14-5.60] 这是一段清晰的中文语音识别测试
```

## Key Features

### One CLI for local multimodal tasks

```bash
modelcli detect photo.jpg --class person --draw-boxes detected.jpg
modelcli ocr screenshot.png --markdown
modelcli asr meeting.wav --lang en --timestamps
modelcli tts "Hello, world." --out hello.wav
```

Every capability shares the same model management, exit codes, progress reporting, and file-safety behavior. After model installation, images, audio, and text remain on the local machine during inference.

### A stable interface for agents

```bash
modelcli --json detect photo.jpg --class person
modelcli --json ocr screenshot.png
modelcli --json asr meeting.wav --lang en
modelcli --json tts "Hello, world." --out /absolute/path/hello.wav
```

In Agent mode, stdout contains exactly one JSON document. Progress and diagnostic messages go only to stderr. Both success and failure use the same envelope format, so callers never need to parse terminal prose.

### Verifiable model caches

ModelCLI generates local manifests for object detection, ASR, and TTS models. Each manifest records the upstream source, requested revision, file size, and SHA-256 digest. `models verify` detects missing, damaged, or modified files. `models install --refresh` downloads and validates a replacement in a temporary cache before publishing it over the installed model.

### Safe output publishing

TTS audio and annotated detection/OCR images refuse to overwrite existing files by default. With `--force`, ModelCLI first writes to a temporary file in the destination directory, validates the result, and then atomically replaces the target. Failures and interruptions leave the previous file intact.

## Command Examples

### Object detection

```bash
# Keep all COCO classes with confidence >= 0.5
modelcli detect photo.jpg

# Repeat --class to keep multiple English COCO class names
modelcli detect street.jpg --confidence 0.6 --class person --class car

# Save an annotated image; --force is required to replace an existing file
modelcli detect street.jpg --draw-boxes detected.jpg --force
```

Object detection always uses PicoDet-L 416 COCO with `CPUExecutionProvider`. It recognizes the 80 fixed COCO classes. It is not an open-vocabulary vision model and is not intended to identify UI elements such as buttons or text fields.

### OCR

```bash
modelcli ocr document.png
modelcli ocr document.png --markdown
modelcli ocr document.png --out result.txt
modelcli ocr document.png --draw-boxes annotated.png
```

### Speech recognition

```bash
modelcli asr recording.wav
modelcli asr recording.wav --lang en --timestamps
modelcli asr recording.wav --lang zh --emotion
modelcli asr recording.wav --no-vad --out transcript.txt
```

`--lang` accepts `auto`, `zh`, `en`, `yue`, `ja`, and `ko`. Silero VAD is enabled by default to split long audio into active speech segments.

### Speech synthesis

```bash
# Human mode writes to ./output.wav when --out is omitted
modelcli tts "Hello, world."
modelcli tts "Hello, world." --out hello.wav
modelcli tts @input.txt --out audiobook.wav
modelcli tts "This is a cloned voice." --prompt-audio my_voice.wav --out cloned.wav
```

When `--prompt-audio` is omitted, ModelCLI uses the default Chinese female prompt installed with the TTS model. `--max-duration` limits generated frames; it is not a wall-clock process timeout.

## Agent JSON Protocol

`--json` is a global option and must appear before the subcommand:

```bash
modelcli --json capabilities
modelcli --json doctor
modelcli --json models list
modelcli --json detect photo.jpg --class person
```

The protocol has the following fixed behavior:

- stdout contains exactly one newline-terminated JSON document; stderr is reserved for progress, dependency logs, and `--debug` tracebacks.
- A missing model returns `MODEL_NOT_INSTALLED`; Agent mode never downloads a model implicitly.
- Add the global `--allow-download` option to explicitly permit a download during inference.
- `models install` and `models install --refresh` are already explicit download actions and do not require `--allow-download`.
- TTS requires an explicit `--out` path, and output paths in the result are absolute.
- The caller owns wall-clock timeouts and SIGTERM/SIGKILL handling. Exit code `124` is reserved for caller-reported timeouts.

Success envelope:

```json
{
  "schema_version": "1",
  "ok": true,
  "operation": "asr",
  "result": {
    "text": "hello",
    "language": "en",
    "segments": []
  },
  "meta": {
    "modelcli_version": "0.3.0",
    "elapsed_ms": 123
  }
}
```

Failure envelope:

```json
{
  "schema_version": "1",
  "ok": false,
  "operation": "ocr",
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Input is not a readable image",
    "retryable": false
  },
  "meta": {
    "modelcli_version": "0.3.0",
    "elapsed_ms": 4
  }
}
```

Exit codes:

| Exit code | Meaning |
| ---: | --- |
| `0` | Success |
| `2` | CLI argument or usage error |
| `3` | Invalid input |
| `4` | Model missing, installation failure, download failure, or verification failure |
| `5` | Inference or internal error |
| `6` | Output conflict or write failure |
| `124` | Reserved for caller-reported timeouts; ModelCLI does not return it itself |
| `130` | SIGINT / Ctrl-C |

Callers should capture stdout, stderr, and the exit code separately. Even for a nonzero exit code, parse the error envelope from stdout. Do not interpret stderr as an application result.

## Model Management

```bash
modelcli models list
modelcli models install detect
modelcli models install asr
modelcli models install tts
modelcli models install all
modelcli models verify all
modelcli models install detect --refresh
modelcli models remove detect
modelcli models prefetch
modelcli models clean
```

`prefetch` is equivalent to `install all`, and `clean` is equivalent to `remove all`. The `all` target processes downloadable capabilities in the order `detect -> asr -> tts`.

| Capability | Model and revision | Approx. size | License |
| --- | --- | ---: | --- |
| Object detection | PicoDet-L 416 COCO, PaddleDetection `release/2.8` | 23.2 MB | Apache 2.0 |
| OCR | PP-OCRv4 mobile | 15 MB | Apache 2.0 |
| ASR | `iic/SenseVoiceSmall-onnx@v2.0.5` INT8 | 242 MB | MIT |
| VAD | Silero VAD | Under 10 MB | MIT |
| TTS | MOSS-TTS-Nano + Audio Tokenizer + prompt | 326 MB | Apache 2.0 |

Installation, refresh, removal, verification, inference, and `doctor --deep` acquire a per-capability cross-process lock. A normal installation never silently updates a verified model. A refresh keeps the old model until its replacement has downloaded and passed validation.

## Diagnostics

```bash
modelcli capabilities
modelcli doctor
modelcli doctor --deep
```

`capabilities` reports CLI/schema versions, commands and options, runtime devices, ONNX providers, the cache directory, and model status. `doctor` checks dependencies, directory writability, model files, manifests, and hashes. `doctor --deep` also loads installed models. Neither command downloads missing models.

## Updating and Uninstalling

Run the installer again to update from `main`. Existing model caches are reused:

```bash
curl -LsSf https://raw.githubusercontent.com/GraySilver/modelcli/main/install.sh | sh
```

Remove the managed source checkout and Python environment:

```bash
curl -LsSf https://raw.githubusercontent.com/GraySilver/modelcli/main/install.sh | sh -s -- --uninstall
```

The uninstaller keeps model caches. Run `modelcli models clean` before uninstalling if you also want to remove them.

## Running from Source

Python 3.11 or 3.12 and [`uv`](https://docs.astral.sh/uv/) are required:

```bash
git clone https://github.com/GraySilver/modelcli.git
cd modelcli
uv sync --frozen
uv run modelcli --help
```

## Development

```bash
git clone https://github.com/GraySilver/modelcli.git
cd modelcli

uv sync --frozen
uv lock --check
uv run pytest -q
uv run python -m compileall -q src
uv build
```

Do not commit models, manifests, generated audio, caches, virtual environments, build artifacts, or temporary files.

## License

ModelCLI is licensed under the [Apache License 2.0](./LICENSE). Individual models and runtime libraries remain subject to their own licenses; review the applicable upstream terms before redistribution or commercial use.
