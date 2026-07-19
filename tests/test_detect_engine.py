from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from modelcli.detect.engine import (
    BoundingBox,
    COCO_LABELS,
    DetectEngine,
    DetectResult,
    Detection,
    load_detection_session,
    parse_detections,
    preprocess_image,
    validate_class_filter,
)
from modelcli.errors import ModelCliError


def test_coco_labels_use_modern_names_and_stable_ids() -> None:
    assert len(COCO_LABELS) == 80
    assert COCO_LABELS[0] == "person"
    assert COCO_LABELS[3] == "motorcycle"
    assert COCO_LABELS[4] == "airplane"
    assert COCO_LABELS[57] == "couch"
    assert COCO_LABELS[62] == "tv"
    assert COCO_LABELS[79] == "toothbrush"


def test_preprocess_matches_official_resize_normalize_and_scale() -> None:
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[:, :, :] = [0, 128, 255]

    blob, scale = preprocess_image(image)

    assert blob.shape == (1, 3, 416, 416)
    assert blob.dtype == np.float32
    assert blob.flags.c_contiguous
    expected_rgb = np.array([1.0, 128 / 255, 0.0], dtype=np.float32)
    expected = (expected_rgb - np.array([0.485, 0.456, 0.406])) / np.array(
        [0.229, 0.224, 0.225]
    )
    np.testing.assert_allclose(blob[0, :, 0, 0], expected, rtol=1e-5)
    np.testing.assert_array_equal(scale, np.array([[208.0, 104.0]], dtype=np.float32))


def test_parse_filters_clips_drops_invalid_boxes_and_sorts() -> None:
    boxes = np.array(
        [
            [2, 0.75, 40.2, 20.1, 120.8, 90.9],
            [0, 0.95, -10, -4, 80.2, 110.7],
            [5, 0.99, 1, 1, 9, 9],
            [0, 0.49, 0, 0, 5, 5],
            [0, 0.8, 20, 20, 10, 10],
        ],
        dtype=np.float32,
    )

    detections = parse_detections(
        boxes,
        np.array([5], dtype=np.int32),
        width=100,
        height=100,
        confidence=0.5,
        classes=("person", "car"),
    )

    assert [(detection.label, detection.bbox) for detection in detections] == [
        ("person", BoundingBox(0, 0, 80, 99)),
        ("car", BoundingBox(40, 20, 99, 91)),
    ]
    assert detections[0].confidence == pytest.approx(0.95)


def test_parse_rejects_invalid_class_id_and_output_shape() -> None:
    with pytest.raises(ModelCliError, match="invalid class id"):
        parse_detections(
            np.array([[80, 0.9, 0, 0, 1, 1]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            width=10,
            height=10,
            confidence=0.5,
            classes=(),
        )

    with pytest.raises(ModelCliError, match="invalid output"):
        parse_detections(
            np.zeros((2, 5), dtype=np.float32),
            np.array([2], dtype=np.int32),
            width=10,
            height=10,
            confidence=0.5,
            classes=(),
        )


def test_unknown_class_has_stable_input_error() -> None:
    with pytest.raises(ModelCliError) as raised:
        validate_class_filter(("person", "automobile"))

    assert raised.value.code == "INVALID_CLASS"
    assert raised.value.exit_code == 3


def test_session_forces_cpu_and_validates_signature(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple] = []

    class FakeOptions:
        log_severity_level = 0

    class FakeSession:
        def __init__(self, path: str, *, sess_options, providers: list[str]) -> None:
            calls.append((path, sess_options.log_severity_level, providers))

        def get_inputs(self):
            return [
                SimpleNamespace(name="image", shape=[None, 3, 416, 416]),
                SimpleNamespace(name="scale_factor", shape=[1, 2]),
            ]

        def get_outputs(self):
            return [SimpleNamespace(shape=[3598, 6]), SimpleNamespace(shape=[1])]

    monkeypatch.setattr("onnxruntime.SessionOptions", FakeOptions)
    monkeypatch.setattr("onnxruntime.InferenceSession", FakeSession)

    session = load_detection_session(tmp_path / "model.onnx")

    assert isinstance(session, FakeSession)
    assert calls == [(str(tmp_path / "model.onnx"), 3, ["CPUExecutionProvider"])]


def test_draw_boxes_writes_atomically_and_preserves_empty_image(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "annotated.png"
    image = np.full((40, 60, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    result = DetectResult(60, 40, 0.5, (), ())

    written = DetectEngine(tmp_path).draw_boxes(source, output, result, force=False)

    assert written == output.resolve()
    np.testing.assert_array_equal(cv2.imread(str(output)), image)


def test_detect_uses_session_outputs_and_returns_original_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.zeros((20, 30, 3), dtype=np.uint8))

    class FakeSession:
        def run(self, outputs, inputs):
            assert outputs is None
            assert inputs["image"].shape == (1, 3, 416, 416)
            np.testing.assert_array_equal(
                inputs["scale_factor"],
                np.array([[20.8, 416 / 30]], dtype=np.float32),
            )
            return (
                np.array([[0, 0.9, 1, 2, 12, 18]], dtype=np.float32),
                np.array([1], dtype=np.int32),
            )

    engine = DetectEngine(tmp_path)
    engine._session = FakeSession()

    result = engine.detect(source, confidence=0.5, classes=("person",))

    assert result.width == 30
    assert result.height == 20
    assert result.detections == (
        Detection(0, "person", pytest.approx(0.9), BoundingBox(1, 2, 12, 18)),
    )
