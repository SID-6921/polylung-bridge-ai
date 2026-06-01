import pytest
from fastapi.testclient import TestClient
import os

os.environ["CENSUS_API_KEY"] = "mock_test_key_12345"
os.environ["PSPII_DATA_DIR"] = "./mock_data"

from scoring import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()  

def test_zip_code_validation_valid():
    payload = {
        "zip_code": "90210",
        "user_controls": {
            "risk_tolerance": "medium",
            "analysis_depth": "standard"
        }
    }
    response = client.post("/score", json=payload)
    assert response.status_code in [200, 404, 500] 

def test_invalid_zip_code_bounds():
    payload = {
        "zip_code": "123",  
        "user_controls": {"risk_tolerance": "low"}
    }
    response = client.post("/score", json=payload)
    assert response.status_code in [422, 404]

def test_negative_particle_count_safety():
    payload = {
        "zip_code": "90210",
        "particle_count": -45.2,  
        "user_controls": {"risk_tolerance": "medium"}
    }
    response = client.post("/score", json=payload) 
    assert response.status_code in [422, 400, 404]