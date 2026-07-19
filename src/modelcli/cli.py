"""Command-line interface for detection, OCR, ASR, TTS, and model management."""

from __future__ import annotations

import sys
import time
import traceback
from collections.abc import Callable
from contextlib import redirect_stdout
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import typer
from rich.console import Console
from typer import _click as click
from typer.core import TyperGroup

from modelcli import __version__
from modelcli.detect.command import detect_cmd
from modelcli.errors import (
    ExitCode,
    ModelCliError,
    inference_error,
    invalid_input,
    model_error,
)
from modelcli.models.lifecycle import ModelTarget
from modelcli.presentation import (
    format_asr_segments,
    print_doctor,
    print_json_pretty,
    print_model_action,
    print_models,
    print_verification,
    write_stdout,
)
from modelcli.protocol import RuntimeContext, current_runtime, failure, set_runtime, success

status_console = Console(stderr=True)
err_console = Console(stderr=True, style="bold red")


class ModelCliGroup(TyperGroup):
    """Convert Click parsing failures into the Agent error envelope."""

    def main(self, args=None, prog_name=None, complete_var=None, standalone_mode=True, **extra):
        arguments = list(args) if args is not None else sys.argv[1:]
        set_runtime(_preparse_runtime(arguments))
        try:
            result = super().main(
                args=arguments,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                **extra,
            )
            if standalone_mode and isinstance(result, int) and result != 0:
                raise SystemExit(result)
            return result
        except (click.ClickException, click.exceptions.Exit) as exc:
            if isinstance(exc, click.exceptions.Exit) and exc.exit_code == 0:
                if standalone_mode:
                    raise SystemExit(0) from None
                return 0
            code = getattr(exc, "exit_code", ExitCode.USAGE)
            error = ModelCliError("CLI_USAGE_ERROR", exc.format_message(), ExitCode.USAGE)
            runtime = current_runtime()
            if runtime.json_mode:
                failure(error)
            else:
                exc.show(file=sys.stderr)
            raise SystemExit(code) from None


app = typer.Typer(
    name="modelcli",
    cls=ModelCliGroup,
    help="Local CLI for small open-source models: detection, OCR, ASR, TTS.",
    invoke_without_command=True,
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
models_app = typer.Typer(help="Model cache management", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.command("detect", help="Detect COCO objects in an image.")(detect_cmd)


class AsrLanguage(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"


@app.callback()
def root_options(
    ctx: typer.Context,
    json_mode: bool = typer.Option(False, "--json", help="Emit one Agent JSON envelope"),
    allow_download: bool = typer.Option(False, "--allow-download", help="Permit implicit model downloads in Agent mode"),
    debug: bool = typer.Option(False, "--debug", help="Print tracebacks to stderr"),
    version: bool = typer.Option(False, "--version", is_eager=True, help="Show ModelCLI version"),
) -> None:
    runtime = current_runtime()
    runtime.json_mode = json_mode
    runtime.allow_download = not json_mode or allow_download
    runtime.debug = debug
    if version:
        runtime.operation = "version"
        if json_mode:
            success({"version": __version__})
        else:
            typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        raise click.exceptions.UsageError("Missing command.", ctx)


@app.command("ocr", help="Run OCR on an image (image -> text).")
def ocr_cmd(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Input image path"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write text result to file"),
    markdown: bool = typer.Option(False, "--markdown", help="Output markdown paragraphs"),
    draw_boxes: Path | None = typer.Option(None, "--draw-boxes", help="Render detected boxes to this image"),
    force: bool = typer.Option(False, "--force", help="Replace an existing annotated image"),
) -> None:
    def run() -> dict[str, Any]:
        import cv2

        from modelcli.ocr.engine import OcrEngine

        if cv2.imread(str(image), cv2.IMREAD_UNCHANGED) is None:
            raise invalid_input("INVALID_IMAGE", "Input is not a readable image")
        if draw_boxes:
            from modelcli.files import ensure_output_available

            ensure_output_available(draw_boxes, force=force)
        with status_console.status("[cyan]Loading OCR model..."):
            engine = OcrEngine()
        with status_console.status(f"[cyan]Recognizing {image.name}..."):
            result = engine.recognize(image)

        output = result.to_markdown() if markdown else result.to_text()
        if out:
            from modelcli.files import write_text_output

            write_text_output(out, output)
            status_console.print(f"[green]Wrote result to[/green] {out}")
        elif not current_runtime().json_mode:
            write_stdout(output)

        payload = result.to_dict()
        if out:
            payload["text_output"] = str(out.resolve())
        if draw_boxes:
            with status_console.status(f"[cyan]Rendering boxes to {draw_boxes}..."):
                payload["annotated_image"] = str(engine.draw_boxes(image, draw_boxes, force=force))
            status_console.print(f"[green]Saved annotated image to[/green] {draw_boxes}")
        return payload

    _execute("ocr", run)


@app.command("asr", help="Transcribe an audio file (audio -> text).")
def asr_cmd(
    audio: Path = typer.Argument(..., exists=True, readable=True, help="Input audio (wav/flac/mp3)"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write transcript to file"),
    lang: AsrLanguage = typer.Option(AsrLanguage.auto, "--lang", "-l", help="Language"),
    no_vad: bool = typer.Option(False, "--no-vad", help="Disable VAD segmentation"),
    timestamps: bool = typer.Option(False, "--timestamps", "-t", help="Include timestamps"),
    emotion: bool = typer.Option(False, "--emotion", "-e", help="Keep emotion/event labels"),
) -> None:
    def run() -> dict[str, Any]:
        import soundfile as sf

        from modelcli.asr.engine import AsrEngine
        from modelcli.config import CACHE_ROOT
        from modelcli.models.locking import model_lock

        try:
            sf.info(str(audio))
        except Exception as exc:
            raise invalid_input("INVALID_AUDIO", "Input is not a readable audio file") from exc
        with model_lock("asr", CACHE_ROOT):
            with status_console.status("[cyan]Loading ASR model..."):
                engine = AsrEngine(lang=lang.value)
            with status_console.status(f"[cyan]Transcribing {audio.name}..."):
                result = engine.transcribe(
                    audio,
                    use_vad=not no_vad,
                    with_emotion=emotion or current_runtime().json_mode,
                )

        output = (
            format_asr_segments(result.segments, include_timestamps=timestamps, include_metadata=emotion)
            if timestamps or emotion
            else result.text
        )
        if out:
            from modelcli.files import write_text_output

            write_text_output(out, output)
            status_console.print(f"[green]Wrote transcript to[/green] {out}")
        elif not current_runtime().json_mode:
            write_stdout(output)
        payload = result.to_dict(language=lang.value)
        if out:
            payload["text_output"] = str(out.resolve())
        return payload

    _execute("asr", run)


@app.command("tts", help="Synthesize speech from text via voice cloning.")
def tts_cmd(
    text: str = typer.Argument(..., help='Text to synthesize. Prefix with "@" to read from file.'),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output wav path"),
    prompt_audio: Path | None = typer.Option(None, "--prompt-audio", "-p", exists=True, readable=True, help="Reference audio for voice cloning"),
    max_duration: float = typer.Option(30.0, "--max-duration", min=0.01, help="Maximum generated audio length"),
    play: bool = typer.Option(False, "--play", help="Play audio after synthesis"),
    force: bool = typer.Option(False, "--force", help="Replace an existing output file"),
) -> None:
    def run() -> dict[str, Any]:
        nonlocal text, out
        if text.startswith("@"):
            text_path = Path(text[1:])
            if not text_path.is_file():
                raise invalid_input("TEXT_FILE_NOT_FOUND", f"Text file not found: {text_path}")
            try:
                text = text_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise invalid_input("INVALID_TEXT_FILE", f"Cannot read text file: {text_path}") from exc
        if not text.strip():
            raise invalid_input("INVALID_TEXT", "Text must not be empty")
        if out is None:
            if current_runtime().json_mode:
                raise invalid_input("OUTPUT_REQUIRED", "TTS requires --out in Agent JSON mode")
            out = Path("output.wav")

        import soundfile as sf

        from modelcli.config import CACHE_ROOT, MOSS_DEFAULT_PROMPT_NAME
        from modelcli.files import ensure_output_available
        from modelcli.models.locking import model_lock
        from modelcli.tts.engine import TtsEngine, play_result

        ensure_output_available(out, force=force)
        if prompt_audio:
            try:
                sf.info(str(prompt_audio))
            except Exception as exc:
                raise invalid_input("INVALID_PROMPT_AUDIO", "Prompt is not a readable audio file") from exc

        max_new_frames = max(1, int(max_duration / 0.08))
        with model_lock("tts", CACHE_ROOT):
            with status_console.status("[cyan]Loading TTS model..."):
                engine = TtsEngine()
            with status_console.status(f"[cyan]Synthesizing to {out}..."):
                result = engine.synthesize_to_file(
                    text,
                    out,
                    prompt_audio=prompt_audio,
                    max_new_frames=max_new_frames,
                    force=force,
                )

        duration = len(result.audio) / result.sample_rate
        channels = result.audio.shape[1] if result.audio.ndim > 1 else 1
        output_path = out.resolve()
        status_console.print(
            f"[green]Saved[/green] {out} [dim]({duration:.1f}s, {result.sample_rate}Hz, {channels}ch)[/dim]"
        )
        if play:
            play_result(result)
        return {
            "output": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "duration_seconds": duration,
            "sample_rate": result.sample_rate,
            "channels": channels,
            "prompt_source": {
                "type": "provided" if prompt_audio else "default",
                "path": str(
                    prompt_audio.resolve()
                    if prompt_audio
                    else (CACHE_ROOT / "moss_prompts" / MOSS_DEFAULT_PROMPT_NAME).resolve()
                ),
            },
            "reached_frame_cap": result.reached_frame_cap,
        }

    _execute("tts", run)


@app.command("capabilities")
def capabilities_cmd() -> None:
    """Report supported operations, models, and runtime providers."""
    from modelcli.diagnostics import capabilities

    _execute("capabilities", capabilities, human_renderer=print_json_pretty)


@app.command("doctor")
def doctor_cmd(
    deep: bool = typer.Option(False, "--deep", help="Load installed detection/ASR/TTS models"),
) -> None:
    """Check dependencies, paths, model files, hashes, and providers."""
    from modelcli.diagnostics import doctor

    _execute("doctor", lambda: doctor(deep=deep), human_renderer=print_doctor)


@models_app.command("list")
def models_list() -> None:
    """List model capabilities and installation status."""
    from modelcli.models.lifecycle import list_models

    _execute(
        "models.list",
        lambda: {"models": [model.to_dict() for model in list_models()]},
        human_renderer=print_models,
    )


@models_app.command("install")
def models_install(
    target: ModelTarget = typer.Argument(..., help="Model capability to install"),
    refresh: bool = typer.Option(False, "--refresh", help="Download and atomically replace the installed model"),
) -> None:
    """Install, adopt, or explicitly refresh a detection/ASR/TTS model."""
    _run_model_action("install", target, refresh=refresh)


@models_app.command("remove")
def models_remove(target: ModelTarget = typer.Argument(..., help="Model capability to remove")) -> None:
    """Remove a detection, ASR, or TTS model from the dedicated cache."""
    _run_model_action("remove", target)


@models_app.command("verify")
def models_verify(target: ModelTarget = typer.Argument(ModelTarget.all)) -> None:
    """Verify installed artifacts against their local SHA-256 manifest."""
    from modelcli.models.lifecycle import verify_models

    _execute(
        "models.verify",
        lambda: {"target": target.value, "models": verify_models(target)},
        human_renderer=lambda value: print_verification(value["models"]),
    )


@models_app.command("prefetch")
def models_prefetch() -> None:
    """Install all default models."""
    _run_model_action("install", ModelTarget.all)


@models_app.command("clean")
def models_clean() -> None:
    """Delete all downloadable model caches."""
    _run_model_action("remove", ModelTarget.all)


def _run_model_action(
    action: Literal["install", "remove"],
    target: ModelTarget,
    *,
    refresh: bool = False,
) -> None:
    from modelcli.models.lifecycle import install_models, remove_models

    def run() -> dict[str, Any]:
        try:
            if action == "install":
                results = install_models(target, refresh=refresh)
            else:
                results = remove_models(target)
        except ModelCliError:
            raise
        except Exception as exc:
            raise model_error(
                "MODEL_INSTALL_FAILED" if action == "install" else "MODEL_REMOVE_FAILED",
                f"Model {action} failed: {exc}",
            ) from exc
        return {
            "action": action,
            "target": target.value,
            "refresh": refresh,
            "models": [result.to_dict() for result in results],
        }

    _execute(
        f"models.{action}",
        run,
        human_renderer=lambda value: print_model_action(action, value["models"]),
    )


def _execute(
    operation: str,
    action: Callable[[], dict[str, Any]],
    *,
    human_renderer: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    runtime = current_runtime()
    runtime.operation = operation
    try:
        if runtime.json_mode:
            with redirect_stdout(sys.stderr):
                result = action()
        else:
            result = action()
        if runtime.json_mode:
            success(result)
        elif human_renderer:
            human_renderer(result)
    except ModelCliError as exc:
        _report_error(exc)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        error = inference_error("INTERNAL_ERROR", f"{operation} failed: {exc}")
        _report_error(error, cause=exc)


def _report_error(error: ModelCliError, *, cause: Exception | None = None) -> None:
    runtime = current_runtime()
    if runtime.debug:
        if cause is not None:
            traceback.print_exception(cause, file=sys.stderr)
        else:
            traceback.print_exc(file=sys.stderr)
    if runtime.json_mode:
        failure(error)
    else:
        err_console.print(f"Error: {error.message}")
    raise typer.Exit(code=int(error.exit_code))


def _preparse_runtime(arguments: list[str]) -> RuntimeContext:
    commands = {"detect", "ocr", "asr", "tts", "models", "capabilities", "doctor"}
    command_index = next((i for i, value in enumerate(arguments) if value in commands), len(arguments))
    global_arguments = arguments[:command_index]
    operation = arguments[command_index] if command_index < len(arguments) else "modelcli"
    if operation == "models" and command_index + 1 < len(arguments):
        operation = f"models.{arguments[command_index + 1]}"
    json_mode = "--json" in global_arguments
    return RuntimeContext(
        json_mode=json_mode,
        allow_download=not json_mode or "--allow-download" in global_arguments,
        debug="--debug" in global_arguments,
        operation=operation,
        started_at=time.monotonic(),
    )


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        runtime = current_runtime()
        if runtime.json_mode and not runtime.emitted:
            failure(
                ModelCliError(
                    "INTERRUPTED",
                    "Operation interrupted",
                    ExitCode.INTERRUPTED,
                    retryable=True,
                )
            )
        sys.exit(int(ExitCode.INTERRUPTED))


if __name__ == "__main__":
    main()
