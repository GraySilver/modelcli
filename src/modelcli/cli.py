"""modelcli — Local CLI for small open-source models (OCR, ASR, TTS)."""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.table import Table

from modelcli.models.lifecycle import ModelTarget

app = typer.Typer(
    name="modelcli",
    help="Local CLI for small open-source models: OCR, ASR, TTS.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Model cache management", no_args_is_help=True)
app.add_typer(models_app, name="models")

console = Console()
status_console = Console(stderr=True)
err_console = Console(stderr=True, style="bold red")


class AsrLanguage(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"


def _fail(msg: str, code: int = 1) -> None:
    err_console.print(f"Error: {msg}")
    raise typer.Exit(code=code)


# ---------------- OCR ----------------

@app.command("ocr", help="Run OCR on an image (image -> text).")
def ocr_cmd(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Input image path"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write text result to file"),
    as_json: bool = typer.Option(False, "--json", help="Output structured JSON (boxes + scores)"),
    markdown: bool = typer.Option(False, "--markdown", help="Output markdown paragraphs"),
    draw_boxes: Path | None = typer.Option(None, "--draw-boxes", help="Render detected boxes to this image"),
) -> None:
    from modelcli.ocr.engine import OcrEngine

    with status_console.status("[cyan]Loading OCR model..."):
        engine = OcrEngine()
    with status_console.status(f"[cyan]Recognizing {image.name}..."):
        result = engine.recognize(image)

    if as_json:
        output = result.to_json()
    elif markdown:
        output = result.to_markdown()
    else:
        output = result.to_text()

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output, encoding="utf-8")
        status_console.print(f"[green]Wrote result to[/green] {out}")
    else:
        _write_stdout(output)

    if draw_boxes:
        with status_console.status(f"[cyan]Rendering boxes to {draw_boxes}..."):
            engine.draw_boxes(image, draw_boxes)
        status_console.print(f"[green]Saved annotated image to[/green] {draw_boxes}")


# ---------------- ASR ----------------

@app.command("asr", help="Transcribe an audio file (audio -> text).")
def asr_cmd(
    audio: Path = typer.Argument(..., exists=True, readable=True, help="Input audio (wav/flac/mp3)"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write transcript to file"),
    lang: AsrLanguage = typer.Option(AsrLanguage.auto, "--lang", "-l", help="Language"),
    no_vad: bool = typer.Option(False, "--no-vad", help="Disable VAD segmentation"),
    timestamps: bool = typer.Option(False, "--timestamps", "-t", help="Include [start-end] timestamps"),
    emotion: bool = typer.Option(False, "--emotion", "-e", help="Keep emotion/event labels"),
) -> None:
    from modelcli.asr.engine import AsrEngine

    with status_console.status("[cyan]Loading ASR model (may download on first run)..."):
        engine = AsrEngine(lang=lang.value)
    with status_console.status(f"[cyan]Transcribing {audio.name}..."):
        result = engine.transcribe(
            audio,
            use_vad=not no_vad,
            with_emotion=emotion,
        )

    if timestamps or emotion:
        output = _format_asr_segments(
            result.segments,
            include_timestamps=timestamps,
            include_metadata=emotion,
        )
    else:
        output = result.text

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output, encoding="utf-8")
        status_console.print(f"[green]Wrote transcript to[/green] {out}")
    else:
        _write_stdout(output)


# ---------------- TTS ----------------

@app.command("tts", help="Synthesize speech from text via voice cloning (text -> audio).")
def tts_cmd(
    text: str = typer.Argument(..., help='Text to synthesize. Prefix with "@" to read from file.'),
    out: Path = typer.Option(Path("output.wav"), "--out", "-o", help="Output wav path"),
    prompt_audio: Path | None = typer.Option(
        None,
        "--prompt-audio",
        "-p",
        exists=True,
        readable=True,
        help="Reference audio for voice cloning. Defaults to bundled Chinese female sample.",
    ),
    max_duration: float = typer.Option(
        30.0,
        "--max-duration",
        min=0.01,
        help="Maximum generated audio length in seconds (caps generated frames).",
    ),
    play: bool = typer.Option(False, "--play", help="Play audio after synthesis (requires simpleaudio)"),
) -> None:
    from modelcli.tts.engine import TtsEngine, play_result

    if text.startswith("@"):
        text_path = Path(text[1:])
        if not text_path.exists():
            _fail(f"Text file not found: {text_path}")
        text = text_path.read_text(encoding="utf-8")

    # 1 frame @ 12.5 Hz token rate -> 80 ms of audio.
    max_new_frames = max(1, int(max_duration / 0.08))

    with status_console.status("[cyan]Loading TTS model (may download on first run)..."):
        engine = TtsEngine()
    with status_console.status(f"[cyan]Synthesizing to {out}..."):
        result = engine.synthesize_to_file(
            text,
            out,
            prompt_audio=prompt_audio,
            max_new_frames=max_new_frames,
        )

    duration = len(result.audio) / result.sample_rate
    channels = result.audio.shape[1] if result.audio.ndim > 1 else 1
    status_console.print(
        f"[green]Saved[/green] {out}  "
        f"[dim]({duration:.1f}s, {result.sample_rate}Hz, {channels}ch, "
        f"{out.stat().st_size / 1024:.0f} KB)[/dim]"
    )

    if play:
        try:
            play_result(result)
        except ImportError as e:
            _fail(str(e))


# ---------------- models ----------------

@models_app.command("list")
def models_list(
    as_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """List model capabilities and their installation status."""
    from modelcli.models.lifecycle import list_models

    models = list_models()
    if as_json:
        _write_json({"models": [model.to_dict() for model in models]})
        return

    table = Table(title="Models")
    table.add_column("Capability", style="cyan")
    table.add_column("Model")
    table.add_column("Status", style="green")
    table.add_column("Size", justify="right")
    for model in models:
        size = _human_size(model.size_bytes) if model.size_bytes else "-"
        table.add_row(model.name, model.model, model.status, size)
    console.print(table)


@models_app.command("install")
def models_install(
    target: ModelTarget = typer.Argument(..., help="Model capability to install"),
    as_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Install and validate an ASR or TTS model."""
    _run_model_action("install", target, as_json=as_json)


@models_app.command("remove")
def models_remove(
    target: ModelTarget = typer.Argument(..., help="Model capability to remove"),
    as_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Remove an ASR or TTS model from the dedicated cache."""
    _run_model_action("remove", target, as_json=as_json)


@models_app.command("prefetch")
def models_prefetch(
    as_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Pre-download all default models."""
    _run_model_action("install", ModelTarget.all, as_json=as_json)


@models_app.command("clean")
def models_clean(
    as_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Delete all downloadable model caches."""
    _run_model_action("remove", ModelTarget.all, as_json=as_json)


def _run_model_action(
    action: Literal["install", "remove"],
    target: ModelTarget,
    *,
    as_json: bool,
) -> None:
    from modelcli.models.lifecycle import install_models, remove_models

    if not as_json:
        verb = "Installing" if action == "install" else "Removing"
        console.print(f"[bold]{verb} {target.value}...[/bold]")

    try:
        results = install_models(target) if action == "install" else remove_models(target)
    except Exception as exc:
        _fail(f"Model {action} failed: {exc}")
    if as_json:
        _write_json(
            {
                "action": action,
                "target": target.value,
                "models": [result.to_dict() for result in results],
            }
        )
        return

    for result in results:
        if action == "install":
            state = "installed" if result.changed else "already installed"
        else:
            state = "removed" if result.changed else "already missing"
        console.print(f"  [green]✓[/green] {result.name}: {state}")


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _write_stdout(output: str) -> None:
    sys.stdout.write(output)
    if output and not output.endswith("\n"):
        sys.stdout.write("\n")


def _write_json(value: dict) -> None:
    _write_stdout(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _format_asr_segments(
    segments: list,
    *,
    include_timestamps: bool,
    include_metadata: bool,
) -> str:
    lines: list[str] = []
    for segment in segments:
        parts: list[str] = []
        if include_timestamps:
            parts.append(f"[{segment.start:.2f}-{segment.end:.2f}]")
        parts.append(segment.text)

        metadata: list[str] = []
        if include_metadata and segment.emotion:
            metadata.append(f"emotion={segment.emotion}")
        if include_metadata and segment.events:
            metadata.append(f"events={'|'.join(segment.events)}")
        if metadata:
            parts.append(f"({', '.join(metadata)})")
        lines.append(" ".join(part for part in parts if part))
    return "\n".join(lines)


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
