# ModelCLI

[![Commit Activity](https://img.shields.io/github/commit-activity/m/GraySilver/modelcli)](https://github.com/GraySilver/modelcli/graphs/commit-activity)
[![Version](https://img.shields.io/badge/version-0.3.0-0A7B83)](https://github.com/GraySilver/modelcli/blob/main/src/modelcli/__init__.py)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://github.com/GraySilver/modelcli/blob/main/pyproject.toml)
[![Agent JSON](https://img.shields.io/badge/Agent%20JSON-v1-2F855A)](#agent-json-%E5%8D%8F%E8%AE%AE)
[![License](https://img.shields.io/github/license/GraySilver/modelcli)](https://github.com/GraySilver/modelcli/blob/main/LICENSE)

**简体中文** | [English](#english)

[快速开始](#快速开始) | [命令示例](#命令示例) | [Agent JSON 协议](#agent-json-协议) | [模型管理](#模型管理) | [开发](#开发)

> 在本地统一运行目标检测、OCR、语音识别和语音合成，为人类和 Agent 提供同一套命令行入口。

ModelCLI 将多个开源小模型封装为一致的 CLI。默认模式适合直接在终端中使用；加上全局 `--json` 后，会输出稳定、可解析的 JSON 信封，方便 JarvisBot、自动化脚本和其他 Agent 通过子进程调用。

模型下载一次后即可离线推理。默认缓存位于 `~/Library/Caches/modelcli`；Linux 等其他平台遵循 [`platformdirs`](https://platformdirs.readthedocs.io/) 的用户缓存目录约定。

| 能力 | 默认模型 | 输入 | 输出 |
| --- | --- | --- | --- |
| 目标检测 | PicoDet-L 416 COCO | 图片 | 80 类物体、置信度、像素坐标、标注图 |
| OCR | PP-OCRv4 mobile | 图片 | 中英文文本、文本行、坐标、置信度、标注图 |
| ASR | SenseVoiceSmall INT8 ONNX | WAV / FLAC / MP3 | 中、英、粤、日、韩文本，时间段、情绪和事件 |
| TTS | MOSS-TTS-Nano | 文本 + 可选参考音频 | 48 kHz 立体声克隆语音 |

## 快速开始

一行安装，适用于 macOS 和 Linux：

```bash
curl -LsSf https://raw.githubusercontent.com/GraySilver/modelcli/main/install.sh | sh
```

安装脚本会在需要时安装 `uv`，使用 Python 3.12 和仓库中的 `uv.lock` 创建隔离环境，并将 `modelcli` 放入 `~/.local/bin`。安装代码和 Python 依赖时不会下载推理模型。

```bash
modelcli --version
modelcli doctor
modelcli models list
```

如果终端提示找不到 `modelcli`，将用户命令目录加入 `PATH`：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

也可以先检查脚本再执行：

```bash
curl -LsSf https://raw.githubusercontent.com/GraySilver/modelcli/main/install.sh -o install.sh
less install.sh
sh install.sh
```

使用 Python 3.11 安装：

```bash
MODELCLI_PYTHON=3.11 sh install.sh
```

首次推理缺少模型时，人类模式会自动下载对应模型。希望提前准备全部模型，可运行：

```bash
modelcli models prefetch
modelcli models verify all
```

目标检测、ASR 和 TTS 模型合计约 590 MB；OCR 与 VAD 随 Python 依赖提供。

## 核心特性

### 一套命令处理本地多模态任务

```bash
modelcli detect photo.jpg --class person --draw-boxes detected.jpg
modelcli ocr screenshot.png --markdown
modelcli asr meeting.wav --lang zh --timestamps
modelcli tts "你好，世界。" --out hello.wav
```

每种能力共享一致的模型管理、错误码、进度输出和文件安全策略。下载完成后，图片、音频和文本都留在本机处理。

### 面向 Agent 的稳定接口

```bash
modelcli --json detect photo.jpg --class person
modelcli --json ocr screenshot.png
modelcli --json asr meeting.wav --lang zh
modelcli --json tts "你好，世界。" --out /absolute/path/hello.wav
```

Agent 模式下 stdout 只包含一个 JSON 文档，进度和诊断信息只写入 stderr。成功与失败都使用同一套信封结构，调用方无需解析终端文案。

### 可验证的模型缓存

ModelCLI 为目标检测、ASR 和 TTS 模型生成本地 manifest，记录模型来源、请求版本、文件大小和 SHA-256。`models verify` 可以发现缺失、损坏或被修改的文件；`models install --refresh` 会在临时缓存中完成下载和验证，再替换现有模型。

### 安全的输出发布

TTS 音频以及目标检测/OCR 标注图默认拒绝覆盖已有文件。传入 `--force` 时，结果先写入同目录临时文件，验证成功后再原子替换目标；失败或中断不会破坏原文件。

## 命令示例

### 目标检测

```bash
# 默认保留置信度不低于 0.5 的全部 COCO 类别
modelcli detect photo.jpg

# --class 可重复，类别名使用英文 COCO 名称
modelcli detect street.jpg --confidence 0.6 --class person --class car

# 保存带检测框的图片；覆盖已有文件时需要 --force
modelcli detect street.jpg --draw-boxes detected.jpg --force
```

目标检测固定使用 PicoDet-L 416 COCO 和 `CPUExecutionProvider`。它只识别 COCO 的 80 个固定类别，不是开放词汇视觉模型，也不用于识别按钮、输入框等 UI 元素。

### OCR

```bash
modelcli ocr document.png
modelcli ocr document.png --markdown
modelcli ocr document.png --out result.txt
modelcli ocr document.png --draw-boxes annotated.png
```

### 语音识别

```bash
modelcli asr recording.wav
modelcli asr recording.wav --lang zh --timestamps
modelcli asr recording.wav --lang en --emotion
modelcli asr recording.wav --no-vad --out transcript.txt
```

`--lang` 支持 `auto`、`zh`、`en`、`yue`、`ja` 和 `ko`。默认启用 Silero VAD，将长音频切分为有效语音段。

### 语音合成

```bash
# 未指定 --out 时，人类模式默认写入 ./output.wav
modelcli tts "你好，世界。"
modelcli tts "你好，世界。" --out hello.wav
modelcli tts @input.txt --out audiobook.wav
modelcli tts "这是一段克隆语音。" --prompt-audio my_voice.wav --out cloned.wav
```

未传 `--prompt-audio` 时使用随 TTS 模型安装的默认中文女声。`--max-duration` 控制生成 frame 上限，不是进程的墙钟超时。

## Agent JSON 协议

`--json` 是全局选项，必须放在子命令之前：

```bash
modelcli --json capabilities
modelcli --json doctor
modelcli --json models list
modelcli --json detect photo.jpg --class person
```

固定行为：

- stdout 始终只有一个以换行结尾的 JSON 文档；stderr 只用于进度、依赖日志和 `--debug` traceback。
- 缺少模型时返回 `MODEL_NOT_INSTALLED`，不会隐式下载。
- 需要在推理期间下载时，显式加入全局 `--allow-download`。
- `models install` 和 `models install --refresh` 本身就是显式下载动作，不需要 `--allow-download`。
- TTS 必须显式传入 `--out`，结果中的输出路径为绝对路径。
- 调用方负责墙钟超时以及 SIGTERM/SIGKILL；退出码 `124` 为调用方超时保留。

成功信封示例：

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

失败信封示例：

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

退出码：

| 退出码 | 含义 |
| ---: | --- |
| `0` | 成功 |
| `2` | CLI 参数或用法错误 |
| `3` | 输入无效 |
| `4` | 模型缺失、安装、下载或校验失败 |
| `5` | 推理或内部错误 |
| `6` | 输出冲突或写入失败 |
| `124` | 调用方超时保留，ModelCLI 自身不返回 |
| `130` | SIGINT / Ctrl-C |

调用方应分别捕获 stdout、stderr 和返回码。即使返回码非零，也应从 stdout 解析同一个错误信封；stderr 不能作为业务结果解析。

## 模型管理

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

`prefetch` 等价于 `install all`，`clean` 等价于 `remove all`。`all` 按 `detect -> asr -> tts` 的顺序处理三个可下载能力。

| 能力 | 模型与版本 | 约大小 | 许可 |
| --- | --- | ---: | --- |
| 目标检测 | PicoDet-L 416 COCO，PaddleDetection `release/2.8` | 23.2 MB | Apache 2.0 |
| OCR | PP-OCRv4 mobile | 15 MB | Apache 2.0 |
| ASR | `iic/SenseVoiceSmall-onnx@v2.0.5` INT8 | 242 MB | MIT |
| VAD | Silero VAD | 10 MB 内 | MIT |
| TTS | MOSS-TTS-Nano + Audio Tokenizer + prompt | 326 MB | Apache 2.0 |

安装、刷新、删除、校验、推理和 `doctor --deep` 按能力使用跨进程锁。普通安装不会静默更新已经校验的模型；`--refresh` 成功前会保留旧模型。

## 诊断

```bash
modelcli capabilities
modelcli doctor
modelcli doctor --deep
```

`capabilities` 报告 CLI/schema 版本、命令和选项、运行设备、ONNX provider、缓存目录及模型状态。`doctor` 默认进行依赖、目录可写性、模型文件、manifest 和 hash 检查；`--deep` 还会实际加载已安装模型。两种诊断都不会下载缺失模型。

## 更新与卸载

重新执行安装脚本即可从 `main` 更新，模型缓存不会重复下载：

```bash
curl -LsSf https://raw.githubusercontent.com/GraySilver/modelcli/main/install.sh | sh
```

卸载代码和 Python 环境：

```bash
curl -LsSf https://raw.githubusercontent.com/GraySilver/modelcli/main/install.sh | sh -s -- --uninstall
```

卸载脚本不会删除模型缓存。如需一并清理，先运行 `modelcli models clean`。

## 从源码运行

需要 Python 3.11 或 3.12，以及 [`uv`](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/GraySilver/modelcli.git
cd modelcli
uv sync --frozen
uv run modelcli --help
```

## 开发

```bash
git clone https://github.com/GraySilver/modelcli.git
cd modelcli

uv sync --frozen
uv lock --check
uv run pytest -q
uv run python -m compileall -q src
uv build
```

模型、manifest、音频、缓存、虚拟环境、构建产物和临时文件不应提交到仓库。

## License

ModelCLI 使用 [Apache License 2.0](./LICENSE)。各模型及运行库仍遵循各自的许可证；分发或商用前请同时核对对应上游项目的条款。

---

## English

[简体中文](#modelcli) | **English**

[Quick Start](#quick-start) | [Examples](#command-examples) | [Agent JSON Protocol](#agent-json-protocol) | [Model Management](#model-management) | [Development](#development)

> Run object detection, OCR, speech recognition, and speech synthesis locally through one command-line interface for both humans and agents.

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
