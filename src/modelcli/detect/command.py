"""CLI command for object detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from modelcli.config import DETECT_DEFAULT_CONFIDENCE
from modelcli.errors import invalid_input
from modelcli.presentation import print_detections
from modelcli.protocol import current_runtime


def detect_cmd(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Input image path"),
    confidence: float = typer.Option(
        DETECT_DEFAULT_CONFIDENCE,
        "--confidence",
        min=0.0,
        max=1.0,
        help="Minimum detection confidence",
    ),
    classes: list[str] | None = typer.Option(
        None,
        "--class",
        help="COCO class name to keep; repeat for multiple classes",
    ),
    draw_boxes: Path | None = typer.Option(
        None,
        "--draw-boxes",
        help="Render detected boxes to this image",
    ),
    force: bool = typer.Option(False, "--force", help="Replace an existing annotated image"),
) -> None:
    from modelcli.cli import _execute, status_console

    def run() -> dict[str, Any]:
        import cv2

        from modelcli.config import CACHE_ROOT
        from modelcli.detect.engine import DetectEngine, validate_class_filter
        from modelcli.models.locking import model_lock

        if cv2.imread(str(image), cv2.IMREAD_COLOR) is None:
            raise invalid_input("INVALID_IMAGE", "Input is not a readable image")
        class_filter = tuple(classes or ())
        validate_class_filter(class_filter)
        if draw_boxes:
            from modelcli.files import ensure_output_available

            ensure_output_available(draw_boxes, force=force)
        with model_lock("detect", CACHE_ROOT):
            with status_console.status("[cyan]Loading detection model..."):
                engine = DetectEngine()
            with status_console.status(f"[cyan]Detecting objects in {image.name}..."):
                result = engine.detect(
                    image,
                    confidence=confidence,
                    classes=class_filter,
                )
        payload = result.to_dict()
        if draw_boxes:
            with status_console.status(f"[cyan]Rendering boxes to {draw_boxes}..."):
                payload["annotated_image"] = str(
                    engine.draw_boxes(image, draw_boxes, result, force=force)
                )
            if not current_runtime().json_mode:
                status_console.print(f"[green]Saved annotated image to[/green] {draw_boxes}")
        return payload

    _execute("detect", run, human_renderer=print_detections)
