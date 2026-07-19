"""Human-readable command output formatting."""

from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def write_stdout(output: str) -> None:
    sys.stdout.write(output)
    if output and not output.endswith("\n"):
        sys.stdout.write("\n")


def format_asr_segments(
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


def print_detections(value: dict[str, Any]) -> None:
    detections = value["detections"]
    if not detections:
        console.print("No objects detected.")
        return
    table = Table(title="Detections")
    for heading in ("Class", "Confidence", "Box (x1, y1, x2, y2)"):
        table.add_column(heading)
    for detection in detections:
        box = detection["bbox"]
        table.add_row(
            detection["label"],
            f"{detection['confidence']:.3f}",
            f"{box['x1']}, {box['y1']}, {box['x2']}, {box['y2']}",
        )
    console.print(table)


def print_models(value: dict[str, Any]) -> None:
    table = Table(title="Models")
    for heading in ("Capability", "Model", "Status", "Manifest", "Size"):
        table.add_column(heading)
    for model in value["models"]:
        table.add_row(
            model["name"],
            model["model"],
            model["status"],
            model["manifest_status"],
            human_size(model["size_bytes"]),
        )
    console.print(table)


def print_model_action(action: str, models: list[dict[str, Any]]) -> None:
    for model in models:
        if action == "install":
            state = "installed" if model["changed"] else "already installed"
        else:
            state = "removed" if model["changed"] else "already missing"
        console.print(f"[green]OK[/green] {model['name']}: {state}")


def print_verification(models: list[dict[str, Any]]) -> None:
    for model in models:
        console.print(
            f"[green]OK[/green] {model['name']}: verified {model['files_checked']} files"
        )


def print_doctor(value: dict[str, Any]) -> None:
    for check in value["checks"]:
        marker = "OK" if check["ok"] else "FAIL"
        color = "green" if check["ok"] else "red"
        console.print(f"[{color}]{marker}[/] {check['name']}: {check['detail']}")


def print_json_pretty(value: dict[str, Any]) -> None:
    console.print_json(json.dumps(value, ensure_ascii=False))


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"
