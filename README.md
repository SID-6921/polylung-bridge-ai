# PolyLung Bridge AI

End-to-end prototype linking polymer-specific microplastic signals to pulmonary inflammation risk.
Built as preliminary-data feasibility infrastructure for R21 grant submission.

---

## Preliminary Data Status (Feasibility Framing)

| PD | Description | Status | Feasibility Result | Artifact |
|----|-------------|--------|--------------------|---------|
| PD-1 | Swin-T binary classifier run on LungHist700 H&E images | pilot complete | Reproducible train/eval pipeline validated. Current 1-epoch CPU run shows majority-class collapse (normal F1=0.0, macro F1=0.439, weighted F1=0.687, accuracy=0.783). Not presented as validated classifier performance. | `evidence/public/pd1/pd1_metrics.json` |
| PD-2 | PSPII derivation from published BALF cytokine data | method complete | Demonstrated one-polymer derivation workflow (PS). Output PSPII=1.0 is a normalized value from the current extraction table, not a cross-polymer maximum claim. | `evidence/public/pd2/pspii_weights_final.json` |
| PD-3 | HBIL clinical cohort validation | data unavailable | HBIL data/results are not available for external release. Fallback: keep PD-3 as planned validation pending data-sharing approval. | `evidence/restricted/hbil/README.md` |
| PD-4 | API integration benchmark against mock PSPII route | pilot complete | 100-call benchmark confirms API contract and overhead (success_rate=1.000, p50=8.79 ms, p95=13.11 ms) for mock endpoint, not full model inference latency. | `evidence/public/pd4/benchmark_result.json` |

---

## Repo Layout

```
polylung-bridge-ai/
|-- backend/            FastAPI services (/health, /analyze, /pspii)
|-- frontend/           Streamlit app
|-- scripts/            Reproducible workflow scripts (PD-1, PD-2, PD-4)
|-- evidence/
|   |-- public/
|   |   |-- pd1/        LungHist700 training evidence
|   |   |-- pd2/        PSPII derivation from Balkrishna 2025 Fig.5
|   |   `-- pd4/        API integration benchmark
|   `-- restricted/
|       `-- hbil/       Policy notice - no protected data committed
|-- claims/             claims_ledger.csv - traceability ledger
|-- governance/         Data-sharing and evidence policy
`-- data/               Runtime weights (synced from evidence/public/pd2/)
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
  --epochs 8 --batch-size 8 --num-workers 0 \
  --pretrained \
  --output evidence/public/pd1/pd1_metrics.json
```
PD-1 reporting should prioritize macro F1, per-class F1, and confusion matrix over accuracy alone.

### PD-1 GitHub-only cloud run (no local training)
- Go to `Actions` -> `PD1 Cloud Train` -> `Run workflow`.
- Provide `dataset_url` as a direct `.zip` or `.tar.gz` download link containing LungHist700 folder structure.
- Optional: set repository secret `LUNGHIST700_URL` and leave input empty.
- Artifacts are uploaded as `pd1-cloud-artifacts` and include:
  - `evidence/public/pd1/pd1_metrics.json`
  - `evidence/public/pd1/dataset_split_summary.json`

### PD-2 (PSPII weights from published cytokine data)
```bash
python scripts/pd2_compute_pspii.py \
  --input evidence/public/pd2/balkrishna_2024_extracted.csv \
  --group PS_microplastic \
  --output-json evidence/public/pd2/pspii_weights_final.json \
  --output-csv evidence/public/pd2/pspii_debug.csv
```
Source: Balkrishna et al., *Biomed. Pharmacother.* 187 (2025) 118122, Fig.5 A-E.
Values are figure-read mean estimates (n=8/group, pg/mL, mean +/- SEM).
Current public evidence includes one polymer (PS) and demonstrates method reproducibility.

### PD-3
Not reproducible externally. HBIL clinical data is under institutional lock.
No raw data or results have been committed to this repository.
Fallback plan for proposal framing: maintain PD-3 as planned validation contingent on data-sharing approval and institutional agreement.

### PD-4 (API benchmark)
```bash
# Start backend first, then:
python scripts/integration_benchmark.py \
  --url http://localhost:8000/pspii \
  --calls 100 --polymer PS \
  --output evidence/public/pd4/benchmark_result.json
```
This benchmark measures API contract and overhead against a mock PSPII route, not full image-model inference latency.

### Microplastics_SEM workflow
The Microplastics_SEM archive includes filename-encoded polymer identities in the original SEM images. For the confirmed-polymer subset, use:
```bash
python scripts/prepare_microplastics_sem_polymers.py \
  --archive .tmp/dataset3.zip \
  --output data/microplastics_sem_split \
  --summary evidence/public/pd1/microplastics_sem_split_summary.json

python scripts/train_swin_imagefolder.py \
  --data-dir data/microplastics_sem_split \
  --epochs 8 --batch-size 8 --num-workers 0 \
  --pretrained \
  --output evidence/public/pd1/microplastics_sem_metrics.json

python scripts/make_microplastics_sem_gradcam.py \
  --image-path data/microplastics_sem_split/val/PS \
  --checkpoint evidence/public/pd1/microplastics_sem_swin_t.pt \
  --output evidence/public/pd1/microplastics_sem_explainability_panel.png

python scripts/evaluate_microplastics_sem_robustness.py \
  --data-dir data/microplastics_sem_split \
  --checkpoint evidence/public/pd1/microplastics_sem_swin_t.pt \
  --output evidence/public/pd1/microplastics_sem_robustness.json
```
The training script converts grayscale SEM images to 3-channel input automatically, so it can fine-tune Swin-T without changing the model backbone. The standardized polymer classes currently captured from the archive are PE, PP, PS, PAN, Polyester, and Acrylates.
The explainability panel is now occlusion-based rather than gradient-only to reduce the banding artifact you noticed.

---

## Priority Next Steps
1. Run a GPU-based PD-1 training cycle with class-balancing and multi-epoch schedule; report macro F1, per-class F1, and confusion matrix.
2. Extend PD-2 extraction/derivation to additional polymers using literature-grounded inputs.
3. Keep PD-4 benchmark and inference latency reporting separated in all grant text.
4. Track PD-3 as planned validation until data-sharing access is approved.

---

## Claims Traceability
See `claims/claims_ledger.csv` - each claim maps to a script, artifact, and approval level.

## Governance
See `governance/data_sharing_policy.md` for public vs restricted artifact policy.
