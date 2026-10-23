
"""Shared pytest fixtures."""
import pytest
from app import app  # This imports the Flask app for testing


@pytest.fixture
def client():
    """Provide a Flask test client."""
    with app.test_client() as client:
        yield client
