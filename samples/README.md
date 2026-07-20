# ModelCLI 样例

**简体中文** | [English](./README_EN.md)

这里的输出均由仓库当前版本的 ModelCLI 实际生成，没有手工修改检测框、OCR 文本或 ASR 转写结果。

## 文件

| 能力 | 输入 | 实际输出 |
| --- | --- | --- |
| 目标检测 | [`images/detect-street.jpg`](./images/detect-street.jpg) | [`results/detect-street-boxes.jpg`](./results/detect-street-boxes.jpg) |
| OCR | [`images/ocr-sign.png`](./images/ocr-sign.png) | [`results/ocr-sign-boxes.png`](./results/ocr-sign-boxes.png)、[`results/ocr-sign.txt`](./results/ocr-sign.txt) |
| ASR | [`audio/asr-zh.wav`](./audio/asr-zh.wav) | [`results/asr-zh.txt`](./results/asr-zh.txt) |
| TTS | 文本：`你好，欢迎使用 Model CLI。本地模型，也能拥有清晰、自然的声音。` | [`audio/tts-zh.wav`](./audio/tts-zh.wav) |

## 复现

在仓库根目录运行。已有输出文件时需要保留示例中的 `--force`。

```bash
uv run modelcli detect samples/images/detect-street.jpg \
  --confidence 0.6 \
  --draw-boxes samples/results/detect-street-boxes.jpg \
  --force

uv run modelcli ocr samples/images/ocr-sign.png \
  --out samples/results/ocr-sign.txt \
  --draw-boxes samples/results/ocr-sign-boxes.png \
  --force

uv run modelcli asr samples/audio/asr-zh.wav \
  --lang zh \
  --timestamps \
  --out samples/results/asr-zh.txt

uv run modelcli tts \
  "你好，欢迎使用 Model CLI。本地模型，也能拥有清晰、自然的声音。" \
  --out samples/audio/tts-zh.wav \
  --max-duration 12 \
  --force
```

样例结果可能因运行库、平台和模型上游内容变化而略有不同。

## 素材来源与许可

### `images/detect-street.jpg`

- 标题：*2023 in Amsterdam - a view in the street Javastraat...*
- 作者：[Fons Heijnsbroek](https://commons.wikimedia.org/wiki/User:FotoDutch)
- 来源：[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:2023_in_Amsterdam_-_a_view_in_the_street_Javastraat_in_the_neighbourhood_Indische_buurt_with_sunlight_of_April;_people_are_walking_and_shopping_-_free_download_photo_in_Dutch_street_photography_by_Fons_Heijnsbroek,_Netherlands,_C.tif)
- 许可：[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
- 本仓库保存 Wikimedia 提供的 960px JPEG 缩略版本；像素内容未修改。检测框版本是 ModelCLI 的衍生输出。

### `images/ocr-sign.png`

由本项目维护者创建，原始文件也是 `tests/fixtures/ocr_test.png`。它与 ModelCLI 项目本身采用相同的 [Apache License 2.0](../LICENSE)。

### `audio/asr-zh.wav`

由当前仓库的 MOSS-TTS-Nano 生成以下文本，再转换为 16 kHz、16-bit、单声道 WAV，用于 SenseVoiceSmall ASR 演示：

> 你好，欢迎使用本地模型。这是一段清晰的中文语音识别测试。

使用的[默认参考音](https://github.com/OpenMOSS/MOSS-TTS-Nano/blob/11619374849c649486584e3b10ed55b176a924ee/assets/audio/zh_1.wav)来自 MOSS-TTS-Nano 上游仓库。本项目锁定的提交 `11619374849c649486584e3b10ed55b176a924ee` 采用 [Apache License 2.0](https://github.com/OpenMOSS/MOSS-TTS-Nano/blob/11619374849c649486584e3b10ed55b176a924ee/LICENSE)，版权声明为 Copyright 2026 OpenMOSS Team, Fudan University, SII and MOSI。

### `audio/tts-zh.wav`

由当前仓库的 MOSS-TTS-Nano 和默认中文参考音生成。文本见上方文件表。许可说明与 `audio/asr-zh.wav` 相同。
