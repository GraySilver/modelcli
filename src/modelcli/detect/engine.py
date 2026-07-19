"""PicoDet-L 416 COCO inference through ONNX Runtime on CPU."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modelcli.config import PICODET_INPUT_SIZE, PICODET_MODEL_NAME
from modelcli.errors import inference_error, invalid_input, model_error, output_error
from modelcli.files import atomic_output_path

COCO_LABELS = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
COCO_CLASS_IDS = {label: class_id for class_id, label in enumerate(COCO_LABELS)}

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_EXPECTED_INPUTS = {
    "image": [None, 3, PICODET_INPUT_SIZE, PICODET_INPUT_SIZE],
    "scale_factor": [1, 2],
}


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": asdict(self.bbox),
        }


@dataclass(frozen=True)
class DetectResult:
    width: int
    height: int
    confidence_threshold: float
    class_filter: tuple[str, ...]
    detections: tuple[Detection, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "confidence_threshold": self.confidence_threshold,
            "class_filter": list(self.class_filter),
            "detections": [detection.to_dict() for detection in self.detections],
        }


class DetectEngine:
    """Detect COCO objects with the fixed CPU PicoDet model."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir
        self._session: Any = None

    def detect(
        self,
        image_path: Path,
        *,
        confidence: float,
        classes: tuple[str, ...] = (),
    ) -> DetectResult:
        import cv2

        validate_class_filter(classes)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise invalid_input("INVALID_IMAGE", "Input is not a readable image")
        height, width = image.shape[:2]
        blob, scale_factor = preprocess_image(image)

        try:
            boxes, count = self._ensure_session().run(
                None,
                {"image": blob, "scale_factor": scale_factor},
            )
        except Exception as exc:
            raise inference_error("DETECTION_FAILED", f"Object detection failed: {exc}") from exc

        detections = parse_detections(
            boxes,
            count,
            width=width,
            height=height,
            confidence=confidence,
            classes=classes,
        )
        return DetectResult(width, height, confidence, classes, detections)

    def draw_boxes(
        self,
        image_path: Path,
        output: Path,
        result: DetectResult,
        *,
        force: bool,
    ) -> Path:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise invalid_input("INVALID_IMAGE", "Input is not a readable image")
        for detection in result.detections:
            color = _class_color(detection.class_id)
            box = detection.bbox
            cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color, 2)
            label = f"{detection.label} {detection.confidence:.2f}"
            text_y = max(18, box.y1 - 6)
            cv2.putText(
                image,
                label,
                (box.x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        with atomic_output_path(output, force=force) as temporary:
            if not cv2.imwrite(str(temporary), image):
                raise output_error(
                    "OUTPUT_WRITE_FAILED",
                    f"Cannot write annotated image: {output}",
                )
        return output.resolve()

    def _ensure_session(self) -> Any:
        if self._session is None:
            model_dir = self._model_dir
            if model_dir is None:
                from modelcli.models.lifecycle import prepare_detect_model

                model_dir = prepare_detect_model(validate=False)
                self._model_dir = model_dir
            self._session = load_detection_session(model_dir / PICODET_MODEL_NAME)
        return self._session


def load_detection_session(model_path: Path) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise model_error("MODEL_LOAD_FAILED", f"Failed to load detection model: {exc}") from exc
    _validate_session_signature(session)
    return session


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    if image.ndim != 3 or image.shape[2] != 3:
        raise invalid_input("INVALID_IMAGE", "Input is not a readable color image")
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (PICODET_INPUT_SIZE, PICODET_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    normalized = (resized / 255.0 - _MEAN) / _STD
    blob = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
    scale_factor = np.array(
        [[PICODET_INPUT_SIZE / height, PICODET_INPUT_SIZE / width]],
        dtype=np.float32,
    )
    return blob, scale_factor


def parse_detections(
    boxes: np.ndarray,
    count: np.ndarray,
    *,
    width: int,
    height: int,
    confidence: float,
    classes: tuple[str, ...],
) -> tuple[Detection, ...]:
    boxes = np.asarray(boxes)
    count = np.asarray(count)
    if boxes.ndim != 2 or boxes.shape[1] != 6 or count.shape != (1,):
        raise inference_error("INVALID_MODEL_OUTPUT", "Detection model returned invalid output")
    allowed_ids = {COCO_CLASS_IDS[label] for label in classes} if classes else None
    detections: list[Detection] = []
    for row in boxes:
        class_value, score, raw_x1, raw_y1, raw_x2, raw_y2 = map(float, row)
        if not np.isfinite(row).all() or score < confidence or class_value < 0:
            continue
        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < len(COCO_LABELS):
            raise inference_error(
                "INVALID_MODEL_OUTPUT",
                f"Detection model returned invalid class id: {class_value}",
            )
        if allowed_ids is not None and class_id not in allowed_ids:
            continue
        x1 = min(max(int(round(raw_x1)), 0), width - 1)
        y1 = min(max(int(round(raw_y1)), 0), height - 1)
        x2 = min(max(int(round(raw_x2)), 0), width - 1)
        y2 = min(max(int(round(raw_y2)), 0), height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                class_id=class_id,
                label=COCO_LABELS[class_id],
                confidence=score,
                bbox=BoundingBox(x1, y1, x2, y2),
            )
        )
    return tuple(sorted(detections, key=lambda detection: -detection.confidence))


def validate_class_filter(classes: tuple[str, ...]) -> None:
    unknown = [label for label in classes if label not in COCO_CLASS_IDS]
    if unknown:
        raise invalid_input(
            "INVALID_CLASS",
            f"Unknown COCO class: {unknown[0]}",
        )


def _validate_session_signature(session: Any) -> None:
    inputs = {value.name: value.shape for value in session.get_inputs()}
    outputs = session.get_outputs()
    if inputs != _EXPECTED_INPUTS:
        raise model_error("MODEL_LOAD_FAILED", "Detection model has an unexpected input signature")
    if len(outputs) != 2:
        raise model_error("MODEL_LOAD_FAILED", "Detection model has an unexpected output signature")
    if outputs[0].shape[-1:] != [6] or outputs[1].shape != [1]:
        raise model_error("MODEL_LOAD_FAILED", "Detection model has an unexpected output signature")


def _class_color(class_id: int) -> tuple[int, int, int]:
    return (
        64 + (class_id * 47) % 192,
        64 + (class_id * 89) % 192,
        64 + (class_id * 137) % 192,
    )
