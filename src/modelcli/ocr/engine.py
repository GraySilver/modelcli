"""OCR engine backed by rapidocr-onnxruntime (PP-OCRv4 mobile, ONNXRuntime)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class OcrLine:
    text: str
    score: float
    box: list[list[int]]  # 4 corner points [[x,y], ...]


@dataclass
class OcrResult:
    lines: list[OcrLine]

    def to_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def to_dict(self) -> dict[str, list[dict]]:
        return {"lines": [asdict(line) for line in self.lines]}

    def to_markdown(self) -> str:
        """Naive markdown output: each line as a paragraph; not real layout recovery."""
        return "\n\n".join(line.text for line in self.lines)


class OcrEngine:
    """Wrapper around RapidOCR with lazy model loading."""

    def __init__(self) -> None:
        self._engine: Any = None

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def recognize(self, image: Path) -> OcrResult:
        engine = self._ensure_engine()
        result, _elapse = engine(str(image))
        lines: list[OcrLine] = []
        if not result:
            return OcrResult(lines=[])
        # rapidocr returns a list of [box, text, score] entries.
        for entry in result:
            box, txt, score = entry[0], entry[1], entry[2]
            box_int = [[int(round(p[0])), int(round(p[1]))] for p in box]
            lines.append(
                OcrLine(
                    text=str(txt).strip(),
                    score=float(score),
                    box=box_int,
                )
            )
        return OcrResult(lines=lines)

    def draw_boxes(self, image: Path, out: Path, *, force: bool = False) -> Path:
        """Render detected boxes onto the image and save to out."""
        import cv2
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR, VisRes

        font_path = _find_cjk_font()
        from modelcli.errors import output_error
        from modelcli.files import atomic_output_path

        engine = RapidOCR()
        result, _ = engine(str(image))
        with atomic_output_path(out, force=force) as temporary:
            if result:
                boxes = [np.array(e[0], dtype=np.float32) for e in result]
                texts = [e[1] for e in result]
                scores = [e[2] for e in result]
                vis = VisRes()
                vis_img = vis.draw_ocr_box_txt(
                    str(image), boxes, texts, scores, font_path=font_path
                )
                written = cv2.imwrite(
                    str(temporary), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                )
            else:
                original = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
                written = original is not None and cv2.imwrite(str(temporary), original)
            if not written:
                raise output_error("OUTPUT_WRITE_FAILED", f"Cannot write annotated image: {out}")
        return out.resolve()


def _find_cjk_font() -> str | None:
    """Try to locate a CJK-capable font on the system."""
    candidates = [
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None
