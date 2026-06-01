import json
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, File, UploadFile, Form
import requests

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

PSPII_FILE_PATH = Path(__file__).resolve().parents[2] / "data" / "pspii_weights_final.json"

EXPOSURE_MULTIPLIER = {
    "inhalation": 1.5,
    "ingestion": 1.0,
    "dermal": 0.5,
}


def load_pspii_weights() -> Dict[str, float]:
    if not PSPII_FILE_PATH.exists():
        return PSPII_WEIGHTS

    data = json.loads(PSPII_FILE_PATH.read_text(encoding="utf-8"))
    parsed: Dict[str, float] = {}
    for polymer, value in data.items():
        parsed[polymer] = float(value)
    return parsed


def get_pspii_weight(polymer_type: str) -> float:
    weights = load_pspii_weights()
    return weights.get(polymer_type, 0.3)


def compute_mpri(polymer_type: str, particle_count: int, exposure_route: str, income_index: float) -> float:
    toxicity = TOXICITY_WEIGHTS.get(polymer_type, 2.0)
    route_mult = EXPOSURE_MULTIPLIER[exposure_route.lower()]
    count_factor = min(particle_count / 100.0, 3.0)
    mpri_raw = toxicity * route_mult * count_factor * income_index
    return round(min(mpri_raw, 25.0), 3)


def compute_pspii(polymer_type: str) -> float:
    value = get_pspii_weight(polymer_type)
    if value > 1.0:
        value = value / 5.0
    return round(min(max(value, 0.0), 1.0), 3)


def compute_bridge_score(mpri: float, pspii: float) -> float:
    score = mpri * pspii * 4.0
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
        "route_multiplier": EXPOSURE_MULTIPLIER[exposure_route],
    }


@app.get("/")
def home():
    return {"message": "PolyLung Module is Active"}


from fastapi import FastAPI, Body 


@app.post("/analyze")
async def calculateRisk(
    polyType: str = Form(...), 
    particleCount: int = Form(...), 
    zipcode: str = Form(...), 
    exposRoute: str = Form(...),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()

    vulnerabilityindex = 1.0  
    
    incomeDisplay = "Not available"
    warningMsg = "None"


    try:
        if len(zipcode) != 5:
            warningMsg = "ZIP code is not valid, setting vulnerability index to baseline 1.0."
            incomeDisplay= "Not available"
        else:
            censusURL = f"https://api.census.gov/data/2024/acs/acs5?get=B19013_001E&for=zip%20code%20tabulation%20area:{zipcode}&key=f3e0d86d28bf98269ac3a0e3381ddec28e09793c"
            censusResponse = requests.get(censusURL).json()
            medianincome = int(censusResponse[1][0])
        
            
            if medianincome < 0:
                incomeDisplay = "Not available"
                warningMsg = "Data for this zip code is suppressed, index set to baseline."
            elif medianincome < 50000:
              vulnerabilityindex = 1.3  
              incomeDisplay = medianincome
            elif medianincome > 90000:
              vulnerabilityindex = 0.8  
              incomeDisplay = medianincome
            else:
              incomeDisplay = medianincome
                
    except Exception:
        warningMsg = "Census network error, falling back to baseline 1.0."
        incomeDisplay = "Not available"
        vulnerabilityindex = 1.0
   
    clean_route = exposRoute.strip().lower()

    calculated_mpri = compute_mpri(polyType, particleCount, clean_route, vulnerabilityindex)
    calculated_pspii = compute_pspii(polyType)
    final_bridge_score = compute_bridge_score(calculated_mpri, calculated_pspii)
    assigned_tier = risk_tier(final_bridge_score)

   
    return {
        "status": "success",
        "polymer_type": polyType,
        "bridge_score": final_bridge_score,
        "risk_tier": assigned_tier,
        "Warning Message": warningMsg,
        "Particle Count": particleCount,
        "Exposure Route": exposRoute,
        "ZIP Code": zipcode,
        "internal_metrics": {
            "MPRI Toxicity Weight": TOXICITY_WEIGHTS.get(polyType, 2.0),
            "PSPII Lung Weight": calculated_pspii,
            "Exposure Multiplier": EXPOSURE_MULTIPLIER.get(clean_route, 1.0),
            "Median Area Income": incomeDisplay,
            "Vulnerability Index": vulnerabilityindex,
            "Calculated MPRI": calculated_mpri
        }
    }