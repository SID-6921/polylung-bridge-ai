import json
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, Body, Form, UploadFile, File, HTTPException
import requests
import os

app = FastAPI()

TOXICITY_WEIGHTS = {
    "PVC": 5.0,
    "PS": 4.0,
    "PU": 4.0,
    "PE": 2.0,
    "PP": 2.0,
    "PET": 2.0,
    "Nylon": 2.0,
    "Acrylic": 2.0,
    "PC": 2.0,
    "ABS": 2.0,
}

PSPII_WEIGHTS = {
    "PVC": 4.5,
    "PS": 3.8,
    "PU": 3.2,
    "PET": 2.1,
    "PE": 1.2,
    "PP": 1.3,
    "Nylon": 1.6,
    "Acrylic": 1.8,
    "PC": 2.0,
    "ABS": 2.2,
}

PSPII_FILE_PATH = Path(__file__).resolve().parent / "data" / "pspii_weights_final.json"


try:
    if not PSPII_FILE_PATH.exists():
        raise FileNotFoundError()
    CACHED_PSPII_WEIGHTS = json.loads(PSPII_FILE_PATH.read_text(encoding="utf-8"))
except Exception:
    CACHED_PSPII_WEIGHTS = {k: round(v / 5.0, 3) for k, v in PSPII_WEIGHTS.items()}


EXPOSURE_MULTIPLIER = {
    "inhalation": 1.5,
    "ingestion": 1.0,
    "dermal": 0.5,
}


def get_pspii_weight(polymer_type: str) -> float:
    return float(CACHED_PSPII_WEIGHTS.get(polymer_type.upper().strip(), 0.0))

def compute_mpri(polymer_type: str, particle_count: int, exposure_route: str, vulnerability_index: float) -> float:
    toxicity = TOXICITY_WEIGHTS.get(polymer_type.upper().strip(), 2.0)
    route_mult = EXPOSURE_MULTIPLIER[exposure_route.lower().strip()]
    count_factor = min(particle_count / 100.0, 3.0)
    
    mpri_raw = toxicity * route_mult * count_factor * vulnerability_index
    return round(min(mpri_raw, 25.0), 3)


def compute_pspii(polymer_type: str) -> float:
    value = get_pspii_weight(polymer_type)
    if value > 1.0:
        value = value / 5.0
    return round(min(max(value, 0.0), 1.0), 3)


def compute_bridge_score(mpri: float, pspii: float, vulnerability_index: float) -> float:
    score = mpri * pspii * vulnerability_index * 10.0
    return round(min(score, 100.0), 3)

def risk_tier(score: float) -> str:
    if score < 15:
        return "Low"
    if score < 35:
        return "Elevated"
    if score < 65:
        return "High"
    return "Critical"


def build_details(polymer_type: str, exposure_route: str) -> Dict[str, float]:
    return {
        "toxicity_weight": TOXICITY_WEIGHTS.get(polymer_type, 2.0),
        "pspii_weight": get_pspii_weight(polymer_type),
        "route_multiplier": EXPOSURE_MULTIPLIER[exposure_route.lower()],
    }


@app.get("/")
def home():
    return {"message": "PolyLung Module is Active"}



@app.post("/analyze")
async def calculateRisk(
    polyType: str = Form(...), 
    particleCount: int = Form(...), 
    zipcode: str = Form(...), 
    exposRoute: str = Form(...),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    
   
    if not zipcode or len(zipcode) != 5 or not zipcode.isdigit():
        raise HTTPException(status_code=400, detail="Invalid ZIP code. Must be exactly 5 digits.")

  
    if particleCount < 0:
        raise HTTPException(status_code=400, detail="Particle count cannot be negative.")

    clean_route = exposRoute.strip().lower()
    if clean_route not in ["ingestion", "inhalation", "dermal"]:
        raise HTTPException(status_code=400, detail="Invalid exposure route. Allowed values: ingestion, inhalation, dermal.")
   

    vulnerabilityindex = 1.0  
    
    incomeDisplay = "Not available"
    warningMsg = "None"


    try:
        if len(zipcode) != 5:
            warningMsg = "ZIP code is not valid, setting vulnerability index to baseline 1.0."
            incomeDisplay = "Not available"
            vulnerabilityindex = 1.0
        else:
            
            census_key = os.getenv("CENSUS_API_KEY")
            
            if not census_key:
                warningMsg = "Census API Key missing from cloud configuration, setting vulnerability index to baseline 1.0."
                vulnerabilityindex = 1.0
                incomeDisplay = "Not available"
            else:
                censusURL = f"https://api.census.gov/data/2024/acs/acs5?get=B19013_001E&for=zip%20code%20tabulation%20area:{zipcode}&key={census_key}"
                
                headers = {"User-Agent": "PolyLungBridgeAI/1.0 (Contact: admin@polylung.org)"}
                
                response = requests.get(censusURL, headers=headers, timeout=5)
                response.raise_for_status()
                censusResponse = response.json()
                
                medianincome = int(censusResponse[1][0])
                
                if medianincome < 0:
                    incomeDisplay = "Not available"
                    warningMsg = "Data for this zip code is suppressed, index set to baseline."
                    vulnerabilityindex = 1.0
                elif medianincome < 50000:
                    vulnerabilityindex = 1.3  
                    incomeDisplay = medianincome
                elif medianincome > 90000:
                    vulnerabilityindex = 0.8  
                    incomeDisplay = medianincome
                else:
                    vulnerabilityindex = 1.0
                    incomeDisplay = medianincome
                
    except Exception as e:
        warningMsg = f"Census processing issue, falling back to baseline 1.0. Details: {str(e)}"
        incomeDisplay = "Not available"
        vulnerabilityindex = 1.0
   
    clean_route = exposRoute.strip().lower()

    calculated_mpri = compute_mpri(polyType, particleCount, clean_route, vulnerabilityindex)
    calculated_pspii = compute_pspii(polyType)
    
    final_bridge_score = compute_bridge_score(calculated_mpri, calculated_pspii, 1.0)
    assigned_tier = risk_tier(final_bridge_score)

   
    return {
        "status": "success",
        "polymer_type": polyType,
        "bridge_score": final_bridge_score,
        "risk_tier": assigned_tier,
        "warning_message": warningMsg,
        "particle_count": particleCount,
        "exposure_route": clean_route,
        "zip_code": zipcode,
        "internal_metrics": {
            "mpri_toxicity_weight": TOXICITY_WEIGHTS.get(polyType.upper().strip(), 2.0),
            "pspii_lung_weight": calculated_pspii,
            "exposure_multiplier": EXPOSURE_MULTIPLIER.get(clean_route, 1.0),
            "median_area_income": incomeDisplay,
            "vulnerability_index": vulnerabilityindex,
            "calculated_mpri": calculated_mpri  # Ensure this points to the variable
        }
    }