# PolyLung Bridge-AI Module

A computational scoring module that calculates polymer-resolved lung risk using the PolyLung Bridge Score (PBS), combining microplastic toxicity, lung deposition, exposure route, and community vulnerability into a single interpretable risk metric.

# Mathematical Derivation of the Bridge Score

The PolyLung Bridge Score is calculated in three steps:

# Step 1: MPRI (Microplastic Risk Index)
MRPI=Toxicity Weight x Exposure Multiplier x Vulnerability index
- **Toxicity Weight** — polymer-specific toxicity score Ex. PVC=5.0, PE=2.0
- **Exposure Multiplier** — inhalation = 1.5, ingestion = 1.0, dermal = 0.5
- **Count Factor** — min(particle_count / 100, 3.0)
- **Vulnerability Index** — derived from median area income via 5 year Census 2024 API (low income = 1.3, high income = 0.8, baseline = 1.0)
- MPRI is capped at 25.0

# Step 2: PSPII (Polymer-Specific Pulmonary Impact Index)
PSPII=lung weight for poymer type

# Step 3: Bridge Score
Bridge Score = MPRI × PSPII × Vulnerability Index × 10.0 (capped at 100.0)

# Bridge Score Interpretation

| Score Range | Risk Tier | Meaning |
|-------------|-----------|---------|
| < 15 | Low | Minimal concern |
| 15 - 35 | Elevated | Moderate concern |
| 35 - 65 | High | Significant concern |
| > 65 | Critical | Severe concern |

---

## Worked Example

**Inputs:**
- Polymer: PVC
- Particle Count: 120
- Exposure Route: inhalation
- ZIP Code: 32501 (median income ~$45,000 → vulnerability index = 1.3)

**Calculation:**
Count Factor = min(120 / 100, 3.0) = 1.2
MPRI = 5.0 × 1.5 × 1.2 × 1.3 = 11.7
PSPII = 0.9
Bridge Score = 11.7 × 0.9 × 1.3 × 10.0 = 136.6 → capped at 100.0