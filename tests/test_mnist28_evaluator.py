import gzip
import struct

import numpy as np
import pytest

from mms_multimodal.mnist28 import _metrics, read_idx_gzip


def test_idx_reader_rejects_wrong_shape_and_reads_labels(tmp_path):
    images = tmp_path / "images.gz"
    labels = tmp_path / "labels.gz"
    images.write_bytes(gzip.compress(struct.pack(">IIII", 2051, 2, 28, 28) + bytes(2 * 28 * 28)))
    labels.write_bytes(gzip.compress(struct.pack(">II", 2049, 2) + bytes([3, 8])))
    assert read_idx_gzip(images, images=True).shape == (2, 28, 28)
    assert read_idx_gzip(labels, images=False).tolist() == [3, 8]
    images.write_bytes(gzip.compress(struct.pack(">IIII", 2051, 2, 32, 32) + bytes(2 * 32 * 32)))
    with pytest.raises(ValueError, match="28x28"):
        read_idx_gzip(images, images=True)


def test_classifier_metrics_report_per_class_errors():
    logits = np.zeros((3, 10), dtype=float)
    logits[0, 1] = 4
    logits[1, 2] = 4
    logits[2, 2] = 4
    result = _metrics(logits, np.array([1, 2, 3], dtype=np.uint8))
    assert result["n"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["confusion_matrix"][3][2] == 1
    assert result["per_class_recall"][3] == 0


def test_digit_model_accepts_native_grayscale():
    torch = pytest.importorskip("torch")
    from mms_multimodal.mnist28 import model

    with torch.inference_mode():
        result = model()(torch.zeros((2, 1, 28, 28)))
    assert result.shape == (2, 10)
