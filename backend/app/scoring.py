from typing import Dict

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

EXPOSURE_MULTIPLIER = {
    "inhalation": 1.5,
    "ingestion": 1.0,
    "dermal": 0.5,
}


def compute_mpri(polymer_type: str, particle_count: int, exposure_route: str, income_index: float) -> float:
    toxicity = TOXICITY_WEIGHTS.get(polymer_type, 2.0)
    route_mult = EXPOSURE_MULTIPLIER[exposure_route]
    count_factor = min(particle_count / 100.0, 3.0)
    mpri_raw = toxicity * route_mult * count_factor * income_index
    return round(min(mpri_raw, 25.0), 3)


def compute_pspii(polymer_type: str) -> float:
    raw = PSPII_WEIGHTS.get(polymer_type, 1.5)
    normalized = raw / 5.0
    return round(min(max(normalized, 0.0), 1.0), 3)


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
        "pspii_weight": PSPII_WEIGHTS.get(polymer_type, 1.5),
        "route_multiplier": EXPOSURE_MULTIPLIER[exposure_route],
    }
