from pydantic import BaseModel, Field
from typing import Dict, Optional


class AnalyzeRequest(BaseModel):
    image_url: Optional[str] = Field(default=None, description="Optional URL for external image")
    exposure_route: str = Field(default="ingestion", pattern="^(ingestion|inhalation|dermal)$")
    income_index: float = Field(default=1.0, ge=0.5, le=1.5)


class AnalyzeResponse(BaseModel):
    polymer_type: str
    confidence: float
    particle_count: int
    mpri: float
    pspii: float
    bridge_score: float
    risk_tier: str
    details: Dict[str, float]
