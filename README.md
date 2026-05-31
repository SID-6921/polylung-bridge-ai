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
PYTHONPATH=backend python -m pytest -q
```

## 4-Week Deliverable Mapping
- PD-1: run `scripts/pd1_train_swin_lunghist700.py` and store metrics in `reports/pd1/pd1_metrics.json`
- PD-2: fill `reports/pd2/balkrishna_2024_extracted.csv`, run `scripts/derive_pspii_weights.py`, and save to `reports/pd2/pspii_weights_final.json`
- PD-3: excluded from public artifacts due HBIL data protection policy
- PD-4: run `scripts/integration_benchmark.py` for 100 calls against `/pspii` and save output in `reports/pd4_integration/benchmark_result.json`

## PD-1 Command
```bash
pip install -r requirements-pd1.txt
python scripts/pd1_train_swin_lunghist700.py --data-dir data/lunghist700_binary --epochs 5 --batch-size 16 --lr 1e-4 --published-baseline 0.0000 --output reports/pd1/pd1_metrics.json
```

## PD-2 Command
```bash
python scripts/derive_pspii_weights.py --input reports/pd2/balkrishna_2024_extracted.csv --output reports/pd2/pspii_weights_final.json
```

## PD-4 Command
```bash
python scripts/integration_benchmark.py --url http://localhost:8000/pspii --calls 100 --polymer PS --output reports/pd4_integration/benchmark_result.json
```

## License
Add your preferred license before external distribution.
