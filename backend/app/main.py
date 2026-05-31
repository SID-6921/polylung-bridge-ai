from random import choice, randint, random

from fastapi import FastAPI

from .schemas import AnalyzeRequest, AnalyzeResponse
from .scoring import build_details, compute_bridge_score, compute_mpri, compute_pspii, risk_tier

app = FastAPI(title="PolyLung Bridge AI API", version="0.1.0")

POLYMERS = ["PE", "PP", "PS", "PET", "PVC", "Nylon", "Acrylic", "PU", "PC", "ABS"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    polymer_type = choice(POLYMERS)
    confidence = round(0.7 + (random() * 0.299), 3)
    particle_count = randint(20, 300)

    mpri = compute_mpri(polymer_type, particle_count, req.exposure_route, req.income_index)
    pspii = compute_pspii(polymer_type)
    bridge_score = compute_bridge_score(mpri, pspii)

    return AnalyzeResponse(
        polymer_type=polymer_type,
        confidence=confidence,
        particle_count=particle_count,
        mpri=mpri,
        pspii=pspii,
        bridge_score=bridge_score,
        risk_tier=risk_tier(bridge_score),
        details=build_details(polymer_type, req.exposure_route),
    )
