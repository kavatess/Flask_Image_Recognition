"""Pytest tests for the Flask routes in app.py.

The machine-learning model is mocked wherever a test only cares about how the
route behaves, so those tests are fast and do not depend on model accuracy.
"""
from io import BytesIO

import pytest
from PIL import Image

import app as app_module


def make_png(size=(32, 32), mode="RGB"):
    """Return an in-memory PNG file of a solid colour, ready for upload."""
    buffer = BytesIO()
    Image.new(mode, size, color=(120, 60, 200)[: len(mode)] if mode != "L" else 128).save(
        buffer, format="PNG"
    )
    buffer.seek(0)
    return buffer


def upload(client, file_obj, filename):
    """POST a file to the prediction route as multipart form data."""
    return client.post(
        "/prediction",
        data={"file": (file_obj, filename)},
        content_type="multipart/form-data",
    )


@pytest.fixture
def mock_prediction(monkeypatch):
    """Replace the model prediction so the route always predicts the digit 7."""
    monkeypatch.setattr(app_module, "predict_result", lambda _img: 7)


# ---------------------------------------------------------------- home page
def test_home_page_loads(client):
    """The home page can be accessed successfully."""
    response = client.get("/")
    assert response.status_code == 200


def test_home_page_contains_upload_form(client):
    """The home page offers a form that posts a file to /prediction."""
    html = client.get("/").get_data(as_text=True)
    assert 'action="/prediction"' in html or "/prediction" in html
    assert 'type="file"' in html


# ------------------------------------------------------------ valid request
@pytest.mark.usefixtures("mock_prediction")
def test_valid_image_returns_prediction(client):
    """A valid image produces a response that shows the predicted digit."""
    response = upload(client, make_png(), "digit.png")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert ">7<" in html
    assert "File cannot be processed." not in html


@pytest.mark.usefixtures("mock_prediction")
def test_valid_jpeg_image_is_accepted(client):
    """Other valid image formats (JPEG) are processed as well as PNG."""
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="JPEG")
    buffer.seek(0)
    html = upload(client, buffer, "digit.jpg").get_data(as_text=True)
    assert ">7<" in html


def test_route_passes_preprocessed_image_to_model(client, monkeypatch):
    """The preprocessed array (not the raw upload) is what reaches the model."""
    received = []

    def fake_predict(img):
        received.append(img.shape)
        return 3

    monkeypatch.setattr(app_module, "predict_result", fake_predict)
    html = upload(client, make_png(), "digit.png").get_data(as_text=True)
    assert received == [(1, 224, 224, 3)]
    assert ">3<" in html


def test_prediction_zero_is_displayed(client, monkeypatch):
    """Edge case: the digit 0 is a valid prediction and must not be shown as an error."""
    monkeypatch.setattr(app_module, "predict_result", lambda _img: 0)
    html = upload(client, make_png(), "zero.png").get_data(as_text=True)
    assert ">0<" in html or "display-4 text-dark font-weight-bold" in html


# ------------------------------------------------------------ missing file
def test_missing_file_field_returns_error_message(client):
    """A request without any uploaded file is handled with a friendly error."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert "File cannot be processed." in response.get_data(as_text=True)


def test_wrong_field_name_returns_error_message(client):
    """A file uploaded under an unexpected form field name is treated as missing."""
    response = client.post(
        "/prediction",
        data={"image": (make_png(), "digit.png")},
        content_type="multipart/form-data",
    )
    assert "File cannot be processed." in response.get_data(as_text=True)


def test_empty_filename_returns_error_message(client):
    """Edge case: the browser sends an empty file part when nothing was selected."""
    response = upload(client, BytesIO(b""), "")
    assert "File cannot be processed." in response.get_data(as_text=True)


# ------------------------------------------------------- invalid / odd input
@pytest.mark.parametrize(
    "content, filename",
    [
        (b"this is not an image", "notes.txt"),
        (b"fake_image_data", "fake.jpg"),
        (b"", "empty.png"),
        (b"\x89PNG\r\n\x1a\n" + b"truncated", "truncated.png"),
    ],
)
def test_invalid_file_content_returns_error_message(client, content, filename):
    """Files that are not decodable images are rejected with the error message."""
    response = upload(client, BytesIO(content), filename)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "File cannot be processed." in html


def test_invalid_file_does_not_reach_model(client, monkeypatch):
    """The model must never be called when preprocessing fails."""
    called = []
    monkeypatch.setattr(app_module, "predict_result", called.append)
    upload(client, BytesIO(b"not an image"), "bad.png")
    assert not called


def test_model_failure_is_reported_as_error(client, monkeypatch):
    """If the model raises, the user still gets the error page instead of a crash."""

    def broken_model(_img):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(app_module, "predict_result", broken_model)
    response = upload(client, make_png(), "digit.png")
    assert response.status_code == 200
    assert "File cannot be processed." in response.get_data(as_text=True)


@pytest.mark.parametrize("mode", ["L", "RGBA"])
@pytest.mark.usefixtures("mock_prediction")
def test_image_without_three_channels_is_reported_as_error(client, mode):
    """Edge case: grayscale and RGBA images cannot be reshaped to 3 channels, so they error."""
    response = upload(client, make_png(mode=mode), "digit.png")
    assert "File cannot be processed." in response.get_data(as_text=True)


@pytest.mark.usefixtures("mock_prediction")
def test_very_small_image_is_resized_and_predicted(client):
    """Edge case: a 1x1 image is upscaled to 224x224 and still predicted."""
    html = upload(client, make_png(size=(1, 1)), "tiny.png").get_data(as_text=True)
    assert ">7<" in html


# ------------------------------------------------------------ HTTP handling
def test_get_on_prediction_route_not_allowed(client):
    """The prediction route only accepts POST."""
    assert client.get("/prediction").status_code == 405


def test_unknown_route_returns_404(client):
    """Unknown URLs return 404."""
    assert client.get("/does-not-exist").status_code == 404
