import pytest
from fastapi.testclient import TestClient
import os
import io

os.environ["CENSUS_API_KEY"] = "mock_test_key_12345"
os.environ["PSPII_DATA_DIR"] = "./mock_data"

from scoring import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()  

def test_zip_code_validation_valid():
    fake_image = io.BytesIO(b"fake image content")
    response = client.post("/analyze", data={
        "polyType": "PVC",
        "particleCount": "120",
        "zipcode": "32501",
        "exposRoute": "ingestion",
    }, files={"file": ("test.png", fake_image, "image/png")})
    assert response.status_code == 200

def test_invalid_zip_code_bounds():
    fake_image = io.BytesIO(b"fake image content")
    response = client.post("/analyze", data={
        "polyType": "PVC",
        "particleCount": "120",
        "zipcode": "123",
        "exposRoute": "ingestion",
    }, files={"file": ("test.png", fake_image, "image/png")})
    assert response.status_code == 400

def test_negative_particle_count_safety():
    fake_image = io.BytesIO(b"fake image content")
    response = client.post("/analyze", data={
        "polyType": "PVC",
        "particleCount": "-10",
        "zipcode": "32501",
        "exposRoute": "ingestion",
    }, files={"file": ("test.png", fake_image, "image/png")})
    assert response.status_code == 400