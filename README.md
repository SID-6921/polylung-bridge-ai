# PolyLung Bridge AI

End-to-end prototype platform linking polymer-specific microplastic signals to pulmonary inflammation risk.

## Repo Organization
- `backend/`: FastAPI services (`/health`, `/analyze`, `/pspii`)
- `frontend/`: Streamlit app
- `scripts/`: reproducible workflow scripts for PD-1, PD-2, PD-4
- `evidence/public/`: reviewer-safe, reproducible outputs
- `evidence/restricted/`: placeholders for restricted artifacts (no protected data committed)
- `claims/`: claim ledger linking each statement to an artifact
- `governance/`: data-sharing and evidence policy

## Current Status
- PD-1: executed on LungHist700 and documented in `evidence/public/pd1/`
- PD-2: workflow ready; final numbers pending real WebPlotDigitizer extraction values
- PD-3: excluded from public artifacts due HBIL protection constraints
- PD-4: 100-call integration benchmark executed and documented in `evidence/public/pd4/`

## Quick Start

### Backend
```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

### Frontend
```bash
pip install -r frontend/requirements.txt
set API_URL=http://localhost:8000
streamlit run frontend/app.py
```

### Tests
```bash
PYTHONPATH=backend python -m pytest -q
```

## Workflow Commands

### PD-1
```bash
python scripts/pd1_prepare_lunghist700_binary.py --source data/raw/LungHist700/data/images --output data/lunghist700_binary --summary evidence/public/pd1/dataset_split_summary.json
python scripts/pd1_train_swin_lunghist700.py --data-dir data/lunghist700_binary --epochs 1 --batch-size 4 --num-workers 0 --lr 1e-4 --output evidence/public/pd1/pd1_metrics.json
```

### PD-2
```bash
python scripts/derive_pspii_weights.py --input evidence/public/pd2/balkrishna_2024_extracted.csv --output evidence/public/pd2/pspii_weights_final.json
python -c "import pathlib; p=pathlib.Path('evidence/public/pd2/pspii_weights_final.json'); d=pathlib.Path('data/pspii_weights_final.json'); d.write_text(p.read_text(encoding='utf-8'), encoding='utf-8')"
```

The PD-2 script rejects all-zero placeholder cytokine tables unless `--allow-placeholder` is explicitly supplied.

### PD-4
```bash
python scripts/integration_benchmark.py --url http://localhost:8000/pspii --calls 100 --polymer PS --output evidence/public/pd4/benchmark_result.json
```

## Latest Verified Outputs
- PD-1 (real LungHist700 run, 1 epoch): `binary_accuracy=0.7826`, `weighted_f1=0.6872`
- PD-1 published-context baseline recorded: multiclass balanced accuracy `0.86` (not directly binary-comparable)
- PD-4 (100 calls, mock PSPII endpoint): `success_rate=1.000`, `latency_ms_p50=8.79`, `latency_ms_p95=13.11`

## Notes
- Legacy `reports/` folders contain `MOVED.md` pointers to the new `evidence/public/` locations.
- Do not commit restricted HBIL data or unapproved HBIL metrics.

## PD-2

Use scripts/pd2_compute_pspii.py with the extracted template in evidence/public/pd2/balkrishna_2024_extracted.csv to run the PD-2 compute pipeline.
