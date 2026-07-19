# modelcli

本地小模型命令行工具 —— 把几个开源小模型封装成统一 CLI，支持 OCR / ASR / TTS。模型下载完成后可以离线运行。

- OCR：图片 → 文本（基于 PP-OCRv4 / RapidOCR，中英混合）
- ASR：音频 → 文本（基于阿里 SenseVoice-Small，中/英/粤/日/韩，带情绪和事件标签）
- TTS：文本 → 音频（基于 MOSS-TTS-Nano，声音克隆模式，48 kHz 立体声，支持 20 种语言）

OCR 和 ASR 使用 ONNXRuntime，VAD 和 TTS 使用 PyTorch，默认在本地 CPU 运行。模型第一次运行时自动下载到专属缓存（Mac：`~/Library/Caches/modelcli`）。

## 安装

前置：Python 3.11、uv（推荐）。

```bash
git clone <this-repo> modelcli && cd modelcli
uv sync
```

安装后会在 `.venv/bin/modelcli` 生成 CLI；也可以 `source .venv/bin/activate` 后直接敲 `modelcli`。

## 使用

```bash
# OCR：识别图片文字
modelcli ocr photo.png                      # 打印到 stdout
modelcli ocr photo.png --json               # 输出带坐标和置信度的 JSON
modelcli ocr photo.png --out result.txt     # 写入文件
modelcli ocr photo.png --draw-boxes vis.png # 画出检测框

# ASR：音频转文字
modelcli asr recording.wav                            # 自动语言检测 + VAD 切段
modelcli asr recording.wav --lang zh --timestamps     # 指定中文 + 时间戳
modelcli asr recording.wav --lang en --emotion        # 英文 + 情绪/事件标签
modelcli asr recording.wav --no-vad --out out.txt     # 不做 VAD，写文件

# TTS：文字转语音（MOSS-TTS-Nano 声音克隆）
modelcli tts "你好世界，今天天气不错。"                       # 默认写到 ./output.wav
modelcli tts "你好世界。" --out hello.wav                   # 指定输出文件
modelcli tts @input.txt --out book.wav                      # 从文件读文本
modelcli tts "你好" --prompt-audio my_voice.wav             # 用自己的参考音频克隆声音
modelcli tts "你好" --play                                   # 合成后自动播放（需 simpleaudio）

# 模型缓存管理
modelcli models list          # 查看所有能力的模型状态
modelcli models list --json   # 机器可读状态
modelcli models install asr   # 安装/校验 SenseVoice ASR
modelcli models install tts   # 安装/校验 MOSS-TTS-Nano（主模型 + 音频 tokenizer + 默认参考音频）
modelcli models install all   # 按 ASR、TTS 顺序安装全部可管理模型
modelcli models remove asr    # 删除 ASR 专属缓存
modelcli models remove tts    # 删除 TTS 专属缓存
modelcli models remove all    # 删除全部可管理模型

# 兼容命令
modelcli models prefetch      # 等价于 models install all
modelcli models clean         # 等价于 models remove all
```

### TTS 说明（MOSS-TTS-Nano）

MOSS-TTS-Nano 是 OpenMOSS 团队开源的 0.1B 参数 TTS 模型，采用**声音克隆**模式工作：

- 默认使用仓库内置的中文女声参考音频（`zh_1.wav`）
- 可用 `--prompt-audio` 传入自己的参考 wav（任意说话人，模型会模仿其音色）
- 输出固定为 48 kHz 立体声
- 支持中文、英文、日文、韩文等 20 种语言
- 长文本会自动分句合成
- `--max-duration` 限制最长生成时长（默认 30 秒）

### 命令行输出契约

`ocr` 和 `asr` 的最终结果写入 stdout，模型加载进度、下载提示和文件写入状态写入 stderr，适合由其他程序直接调用：

```bash
/Users/jarvis/Documents/Coding/modelcli/.venv/bin/modelcli asr recording.wav \
  > transcript.txt 2> modelcli.log
```

使用 `--out` 时最终结果只写入指定文件，stdout 为空。`--emotion` 会按分段输出可读标签，例如：

```text
hello world (emotion=neutral, events=Speech|Applause)
```

`install` 和 `remove` 的目标只接受 `asr`、`tts` 或 `all`。OCR 和 VAD 随 Python 包安装，在 `models list` 中显示为 `bundled`，不会被模型缓存命令删除。

模型删除不需要交互确认，重复删除会成功返回 `already missing`。ASR/TTS 推理保留自动下载行为：模型删除后再次执行对应推理命令，会自动重新安装。不要在模型正在推理或安装时并发执行 `remove`。

`remove all` 和兼容命令 `clean` 只清理 ModelCLI 专属的 ASR/TTS 缓存。旧版本写入 `~/.cache/huggingface` 的共享缓存和历史孤立文件不会被迁移或删除。

模型管理命令支持 `--json`。例如 `models list --json` 输出固定的 `asr`、`tts`、`ocr`、`vad` 状态；`install/remove --json` 输出动作、目标、每个模型的状态、是否发生改动和当前大小。JSON 写入 stdout，下载进度和错误写入 stderr。

## 模型与依赖

| 能力 | 模型 | 大小 | 来源 |
|---|---|---|---|
| OCR | PP-OCRv4 mobile（rapidocr-onnxruntime） | ~15 MB | pip 包自带 |
| ASR | SenseVoice-Small | ~1.8 GB（含源模型和 ONNX） | ModelScope `iic/SenseVoiceSmall`（首次运行准备 ONNX） |
| VAD | Silero VAD | <10 MB | silero-vad pip 包自带 |
| TTS | MOSS-TTS-Nano 主模型 + MOSS-Audio-Tokenizer-Nano + 默认中文参考音频 | ~310 MB | ModelScope `OpenMOSS/MOSS-TTS-Nano` + `openmoss/MOSS-Audio-Tokenizer-Nano` |

## 开发与验证

```bash
uv sync                 # 安装依赖
.venv/bin/modelcli --help
.venv/bin/python -m pytest -q
```

## 许可

本项目是 CLI 封装代码，具体模型许可请见各上游仓库：
- RapidOCR: Apache 2.0
- SenseVoice: MIT
- Silero VAD: MIT
- MOSS-TTS-Nano: Apache 2.0
