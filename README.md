# PolyLung Bridge AI

End-to-end prototype linking polymer-specific microplastic signals to pulmonary inflammation risk.
Built as preliminary data infrastructure for R21 grant submission.

---

## Preliminary Data Status

| PD | Description | Status | Key Result | Artifact |
|----|-------------|--------|------------|---------|
| PD-1 | Swin-T binary classifier on LungHist700 H&E images | ? verified | binary_accuracy = 0.7826, weighted_f1 = 0.6872 (1 epoch, CPU) | `evidence/public/pd1/pd1_metrics.json` |
| PD-2 | PSPII index from published BALF cytokine data | ? verified | PS PSPII weight = 1.0 (TNF-a, IL-1ß, IL-5, IL-6, MIP-2a; Balkrishna et al. 2025, Fig.5, n=8) | `evidence/public/pd2/pspii_weights_final.json` |
| PD-3 | HBIL clinical cohort validation | ? data unavailable | HBIL has declined to share data or results publicly; no metric available for external release | `evidence/restricted/hbil/README.md` |
| PD-4 | API integration latency benchmark | ? verified | success_rate = 1.000, p50 = 8.79 ms, p95 = 13.11 ms (100 calls) | `evidence/public/pd4/benchmark_result.json` |

---

## Repo Layout

```
polylung-bridge-ai/
+-- backend/            FastAPI services (/health, /analyze, /pspii)
+-- frontend/           Streamlit app
+-- scripts/            Reproducible workflow scripts (PD-1, PD-2, PD-4)
+-- evidence/
¦   +-- public/
¦   ¦   +-- pd1/        LungHist700 training evidence
¦   ¦   +-- pd2/        PSPII derivation from Balkrishna 2025 Fig.5
¦   ¦   +-- pd4/        API integration benchmark
¦   +-- restricted/
¦       +-- hbil/       Policy notice — no protected data committed
+-- claims/             claims_ledger.csv — traceability ledger
+-- governance/         Data-sharing and evidence policy
+-- data/               Runtime weights (synced from evidence/public/pd2/)
```

---

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

---

## Reproducing Preliminary Data

### PD-1 (LungHist700 classifier)
```bash
python scripts/pd1_prepare_lunghist700_binary.py \
  --source data/raw/LungHist700/data/images \
  --output data/lunghist700_binary \
  --summary evidence/public/pd1/dataset_split_summary.json

python scripts/pd1_train_swin_lunghist700.py \
  --data-dir data/lunghist700_binary \
  --epochs 1 --batch-size 4 --num-workers 0 \
  --output evidence/public/pd1/pd1_metrics.json
```

### PD-2 (PSPII weights from published cytokine data)
```bash
python scripts/pd2_compute_pspii.py \
  --input evidence/public/pd2/balkrishna_2024_extracted.csv \
  --group PS_microplastic \
  --output-json evidence/public/pd2/pspii_weights_final.json \
  --output-csv evidence/public/pd2/pspii_debug.csv
```
Source: Balkrishna et al., *Biomed. Pharmacother.* 187 (2025) 118122, Fig.5 A-E.
Values are figure-read mean estimates (n=8/group, pg/mL, mean ± SEM).

### PD-3
Not reproducible externally. HBIL clinical data is under institutional lock.
No raw data or results have been committed to this repository.

### PD-4 (API benchmark)
```bash
# Start backend first, then:
python scripts/integration_benchmark.py \
  --url http://localhost:8000/pspii \
  --calls 100 --polymer PS \
  --output evidence/public/pd4/benchmark_result.json
```

---

## Claims Traceability
See `claims/claims_ledger.csv` — each claim maps to a script, artifact, and approval level.

## Governance
See `governance/data_sharing_policy.md` for public vs restricted artifact policy.
