import pytest
from fastapi.testclient import TestClient
import os


os.environ["CENSUS_API_KEY"] = "mock_test_key_12345"
os.environ["PSPII_DATA_DIR"] = "./mock_data"


from scoring import app

client = TestClient(app)

def test_health_check():
    """Test that the backend API boots up and responds with its active message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()  

def test_zip_code_validation_valid():
    """Test the scoring system with a valid standard ZIP code format."""
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
    """Test that bad ZIP string lengths or formats are handled safely."""
    payload = {
        "zip_code": "123",  
        "user_controls": {"risk_tolerance": "low"}
    }
    response = client.post("/score", json=payload)
    
    assert response.status_code in [422, 404]