# modelcli

本地小模型命令行工具，把 OCR、ASR、TTS 封装为统一 CLI。既可供人直接使用，也提供稳定的 Agent JSON 协议供 JarvisBot 等调用方通过子进程执行。

- OCR：PP-OCRv4 / RapidOCR，中英混合图片识别，支持文本行、坐标、置信度和标注图
- ASR：SenseVoiceSmall INT8 ONNX，中/英/粤/日/韩，支持 VAD、时间段、情绪和事件
- TTS：MOSS-TTS-Nano 声音克隆，固定 48 kHz 立体声

模型安装后可离线运行。默认缓存位于 `~/Library/Caches/modelcli`（其他平台遵循 `platformdirs`）。

## 安装

要求 Python 3.11 或 3.12，以及 `uv`：

```bash
git clone <this-repo> modelcli
cd modelcli
uv sync --frozen
```

CLI 位于 `.venv/bin/modelcli`：

```bash
.venv/bin/modelcli --version
.venv/bin/modelcli --help
```

## 人类模式

人类模式是默认模式。推理缺少模型时会自动下载；状态和进度写 stderr，主要文本结果写 stdout。

```bash
# OCR
modelcli ocr photo.png
modelcli ocr photo.png --out result.txt
modelcli ocr photo.png --markdown
modelcli ocr photo.png --draw-boxes annotated.png
modelcli ocr photo.png --draw-boxes annotated.png --force

# ASR
modelcli asr recording.wav
modelcli asr recording.wav --lang zh --timestamps
modelcli asr recording.wav --lang en --emotion
modelcli asr recording.wav --no-vad --out transcript.txt

# TTS；人类模式未指定 --out 时默认写 ./output.wav
modelcli tts "你好世界。"
modelcli tts "你好世界。" --out hello.wav
modelcli tts @input.txt --out book.wav
modelcli tts "你好" --prompt-audio my_voice.wav --out cloned.wav
modelcli tts "你好" --out hello.wav --force
```

TTS 的 `--max-duration` 是生成 frame 上限，不是墙钟超时。默认参考音是 MOSS-TTS-Nano 官方中文女声；`--prompt-audio` 可指定自己的参考音频。

## Agent 模式

全局 `--json` 开启 Agent 协议。该选项必须放在子命令之前：

```bash
modelcli --json capabilities
modelcli --json doctor
modelcli --json ocr photo.png
modelcli --json asr recording.wav --lang zh
modelcli --json tts "你好世界。" --out /absolute/path/speech.wav
modelcli --json models list
```

Agent 模式具有以下固定行为：

- 成功和失败都只向 stdout 写一个 JSON 文档，并以换行结束。
- 进度、依赖库输出和 `--debug` traceback 只写 stderr。
- 缺少模型时默认返回 `MODEL_NOT_INSTALLED`，不会隐式下载。
- 使用全局 `--allow-download` 可显式允许推理期间下载：`modelcli --json --allow-download asr input.wav`。
- `models install` 和 `models install --refresh` 本身就是显式下载动作，不需要 `--allow-download`。
- TTS 必须显式传 `--out`，返回值中的输出路径为绝对路径。
- ModelCLI 不实现推理墙钟超时。调用方负责 timeout、SIGTERM/SIGKILL；退出码 `124` 为调用方超时保留。

成功信封：

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
    "modelcli_version": "0.2.0",
    "elapsed_ms": 123
  }
}
```

失败信封：

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
    "modelcli_version": "0.2.0",
    "elapsed_ms": 4
  }
}
```

退出码：

| 退出码 | 含义 |
|---:|---|
| `0` | 成功 |
| `2` | CLI 参数或用法错误 |
| `3` | 输入无效 |
| `4` | 模型缺失、安装、下载或校验失败 |
| `5` | 推理或内部错误 |
| `6` | 输出冲突或写入失败 |
| `124` | 调用方超时保留，ModelCLI 自身不返回 |
| `130` | SIGINT / Ctrl-C |

旧的 `ocr --json`、`models list --json` 等子命令局部选项已删除。Agent 必须使用全局形式 `modelcli --json COMMAND ...`。

### JarvisBot 子进程调用

JarvisBot 直接运行命令行时，应分别捕获 stdout、stderr 和返回码，并为整个进程设置墙钟超时。例如：

```bash
/Users/jarvis/Documents/Coding/modelcli/.venv/bin/modelcli \
  --json asr /absolute/path/input.wav --lang auto
```

stdout 按单个 JSON 文档解析；非零返回码仍要解析同一个错误信封。stderr 只用于日志和诊断，不能当作业务结果。超时后由 JarvisBot 终止进程并对上层报告保留码 `124`。

## 模型管理

```bash
modelcli models list
modelcli models install asr
modelcli models install tts
modelcli models install all
modelcli models verify asr
modelcli models verify tts
modelcli models verify all
modelcli models install asr --refresh
modelcli models remove asr
modelcli models remove tts
modelcli models remove all
```

`models prefetch` 等价于 `models install all`，`models clean` 等价于 `models remove all`。

每个已安装的 ASR/TTS 能力有本地 manifest，记录请求的 ModelScope revision、安装时间、ModelCLI 版本、相对路径、大小和 SHA-256。ASR 固定请求 `v2.0.5`，TTS 请求 `master`；manifest 不虚构不可变的上游提交，本地 SHA-256 集合才是已安装内容的事实源。

- 符合当前模型标准且文件完整、但缺少 manifest 的缓存，会先通过真实加载验证，再在本地补建 manifest。
- 普通 install 不会静默更新已有、已校验的模型。
- `--refresh` 在隔离临时 cache 中下载、加载、生成 manifest，验证成功后再替换；失败保留旧模型。
- TTS 主模型、音频 tokenizer 和默认 prompt 是一个更新单元。
- install、refresh、remove、verify、推理和 `doctor --deep` 按能力使用跨进程锁，最长等待 30 秒；超时返回可重试错误 `MODEL_BUSY`。
- MOSS-TTS-Nano Python 依赖和默认 prompt URL 固定到提交 `11619374849c649486584e3b10ed55b176a924ee`；默认 prompt 下载后校验 SHA-256。
- ASR 固定使用 `iic/SenseVoiceSmall-onnx@v2.0.5` 的量化模型。运行库需要但 ONNX 仓未包含的官方 SentencePiece 文件从原模型仓下载，并校验固定 SHA-256。

`remove` 只删除 ModelCLI 专属的目标模型、prompt 和 manifest，不删除其他缓存。旧版 `iic__SenseVoiceSmall` 缓存不会自动迁移或删除，`remove asr` 也只管理当前量化模型。重复 remove 成功并报告 `already missing`。

## 诊断

```bash
modelcli capabilities
modelcli doctor
modelcli doctor --deep
```

`capabilities` 报告 CLI/schema 版本、命令和选项、OCR/ASR/TTS 能力、Python/平台/设备/CUDA/ONNX provider、cache 目录、模型状态和 Agent 下载/超时策略。

`doctor` 默认只做静态检查：依赖、cache/temp/当前输出目录可写性、模型必需文件、manifest/hash 和运行设备信息。`doctor --deep` 额外加载已安装的 ASR/TTS。两种模式都不会下载缺失模型。

## 输出安全

TTS 音频和 OCR 标注图默认拒绝覆盖现有文件，冲突返回 `OUTPUT_EXISTS`。传 `--force` 时，ModelCLI 先写同目录隐藏临时文件，验证完成后使用原子替换发布；错误或 SIGINT 会清理临时文件并保留原目标。

文本输出由 `--out` 指定时会写文件；OCR/ASR Agent 结果仍完整包含在 JSON 信封中。

## 模型与依赖

| 能力 | 模型 | 约大小 | 来源 |
|---|---|---:|---|
| OCR | PP-OCRv4 mobile | 15 MB | Python 包内置 |
| ASR | SenseVoiceSmall INT8 ONNX | 约 242 MB | ModelScope `iic/SenseVoiceSmall-onnx@v2.0.5` |
| VAD | Silero VAD | 10 MB 内 | Python 包内置 |
| TTS | MOSS-TTS-Nano + Audio Tokenizer + prompt | 310 MB | ModelScope / 固定提交 prompt |

OCR 和 ASR 使用 ONNXRuntime，VAD 和 TTS 使用 PyTorch。ModelScope 单文件 HTTP 下载超时由 `MODELSCOPE_DOWNLOAD_TIMEOUT` 控制；ModelCLI 未设置时默认使用 120 秒。

## 开发与验证

```bash
uv lock --check
uv sync --frozen
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src
uv build
```

模型、manifest、音频、cache、虚拟环境、构建产物和临时文件不得提交到仓库。

## 模型许可

- RapidOCR：Apache 2.0
- SenseVoice：MIT
- Silero VAD：MIT
- MOSS-TTS-Nano：Apache 2.0
