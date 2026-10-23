"""Pytest unit tests for the preprocessing and prediction helpers in model.py.

``model.model.predict`` is replaced by a stub, so no real inference is run.
"""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import model as model_module
from model import preprocess_img, predict_result


def image_file(size=(50, 80), mode="RGB", color=(255, 0, 0), fmt="PNG"):
    """Return an in-memory image file object."""
    buffer = BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt)
    buffer.seek(0)
    return buffer


class StubModel:  # pylint: disable=too-few-public-methods
    """Minimal stand-in for a Keras model returning fixed class probabilities."""

    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.received = None

    def predict(self, batch):
        """Record the batch and return the canned probabilities."""
        self.received = batch
        return np.array([self.probabilities])


# --------------------------------------------------------- preprocess_img
def test_preprocess_output_shape_and_type():
    """The output is a float array of shape (1, 224, 224, 3)."""
    result = preprocess_img(image_file())
    assert result.shape == (1, 224, 224, 3)
    assert np.issubdtype(result.dtype, np.floating)


@pytest.mark.parametrize("size", [(1, 1), (224, 224), (500, 100), (100, 500)])
def test_preprocess_resizes_any_dimension(size):
    """Edge case: tiny, exact, wide and tall images are all resized to 224x224."""
    assert preprocess_img(image_file(size=size)).shape == (1, 224, 224, 3)


def test_preprocess_normalises_pixels_to_unit_range():
    """Pixel values are scaled from 0-255 to 0-1."""
    white = preprocess_img(image_file(color=(255, 255, 255)))
    black = preprocess_img(image_file(color=(0, 0, 0)))
    assert white.max() == pytest.approx(1.0)
    assert black.max() == pytest.approx(0.0)


def test_preprocess_keeps_colour_channels_in_order():
    """A pure red image keeps red in channel 0 and nothing in channels 1 and 2."""
    result = preprocess_img(image_file(color=(255, 0, 0)))
    assert result[0, :, :, 0].min() == pytest.approx(1.0)
    assert result[0, :, :, 1:].max() == pytest.approx(0.0)


def test_preprocess_accepts_file_path(tmp_path):
    """preprocess_img also accepts a path on disk, not only a file object."""
    path = tmp_path / "digit.png"
    path.write_bytes(image_file().getvalue())
    assert preprocess_img(str(path)).shape == (1, 224, 224, 3)


def test_preprocess_missing_path_raises():
    """A path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("invalid/path/to/image.jpeg")


def test_preprocess_non_image_data_raises():
    """Data that is not an image raises PIL's UnidentifiedImageError."""
    with pytest.raises(Image.UnidentifiedImageError):
        preprocess_img(BytesIO(b"definitely not an image"))


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_preprocess_rejects_images_without_three_channels(mode):
    """Edge case: only 3-channel images can be reshaped; grayscale/RGBA raise ValueError."""
    with pytest.raises(ValueError):
        preprocess_img(image_file(mode=mode, color=128 if mode == "L" else (1, 2, 3, 4)))


# ---------------------------------------------------------- predict_result
def test_predict_result_returns_class_with_highest_probability(monkeypatch):
    """The predicted digit is the index of the highest probability."""
    probabilities = [0.01] * 10
    probabilities[4] = 0.91
    monkeypatch.setattr(model_module, "model", StubModel(probabilities))
    assert predict_result(np.zeros((1, 224, 224, 3))) == 4


def test_predict_result_returns_integer(monkeypatch):
    """The prediction is an integer class index."""
    monkeypatch.setattr(model_module, "model", StubModel([0.2, 0.8]))
    assert isinstance(predict_result(np.zeros((1, 224, 224, 3))), (int, np.integer))


def test_predict_result_passes_input_to_model(monkeypatch):
    """The preprocessed batch is handed to the model unchanged."""
    stub = StubModel([0.5, 0.5])
    monkeypatch.setattr(model_module, "model", stub)
    batch = np.ones((1, 224, 224, 3))
    predict_result(batch)
    assert stub.received is batch


def test_predict_result_tie_returns_first_index(monkeypatch):
    """Edge case: with equal probabilities the lowest index wins."""
    monkeypatch.setattr(model_module, "model", StubModel([0.25, 0.25, 0.25, 0.25]))
    assert predict_result(np.zeros((1, 224, 224, 3))) == 0


def test_predict_result_first_and_last_class(monkeypatch):
    """Edge case: the boundary classes 0 and 9 are both returned correctly."""
    monkeypatch.setattr(model_module, "model", StubModel([1.0] + [0.0] * 9))
    assert predict_result(np.zeros((1, 224, 224, 3))) == 0
    monkeypatch.setattr(model_module, "model", StubModel([0.0] * 9 + [1.0]))
    assert predict_result(np.zeros((1, 224, 224, 3))) == 9
