# PolyLung Bridge AI

End-to-end prototype platform linking polymer-specific microplastic signals to a pulmonary inflammation risk score.

## What Is Included
- FastAPI backend with `/health` and `/analyze`
- Scoring engines for MPRI and PSPII plus Bridge Score tiering
- Streamlit frontend for interactive analysis
- Docker + docker-compose for one-command local runtime
- GitHub Actions CI (pytest)
- Sample validation dataset and model card

## Quick Start (Local)

### 1) Backend
```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

### 2) Frontend
```bash
pip install -r frontend/requirements.txt
set API_URL=http://localhost:8000
streamlit run frontend/app.py
```

## Docker Run
```bash
docker compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

## API Contract

### `GET /health`
Returns service status.

### `POST /analyze`
Request body:
```json
{
  "exposure_route": "ingestion",
  "income_index": 1.0
}
```

Response includes:
- `polymer_type`
- `confidence`
- `particle_count`
- `mpri`
- `pspii`
- `bridge_score`
- `risk_tier`
- `details`

## Test
```bash
PYTHONPATH=backend pytest -q
```

## 4-Week Deliverable Mapping
- PD-1: attach trained Swin results and baseline comparison in `/reports/pd1`
- PD-2: store extracted cytokine values and normalization notebook in `/reports/pd2`
- PD-3: add HBIL 3-paragraph writeup in `/reports/pd3_hbil_writeup.md`
- PD-4: run 100-call integration benchmark and save logs in `/reports/pd4_integration`

## License
Add your preferred license before external distribution.
