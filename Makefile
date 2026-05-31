PYTHON=python

install-backend:
	pip install -r backend/requirements.txt

install-frontend:
	pip install -r frontend/requirements.txt

run-backend:
	uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000

run-frontend:
	set API_URL=http://localhost:8000 && streamlit run frontend/app.py

test:
	PYTHONPATH=backend python -m pytest -q

compose-up:
	docker compose up --build

pd2-derive:
	python scripts/derive_pspii_weights.py --input evidence/public/pd2/balkrishna_2024_extracted.csv --output evidence/public/pd2/pspii_weights_final.json

pd2-sync-weights:
	python -c "import json, pathlib; p=pathlib.Path('evidence/public/pd2/pspii_weights_final.json'); d=pathlib.Path('data/pspii_weights_final.json'); d.write_text(p.read_text(encoding='utf-8'), encoding='utf-8')"

pd4-benchmark:
	python scripts/integration_benchmark.py --url http://localhost:8000/pspii --calls 100 --polymer PS --output evidence/public/pd4/benchmark_result.json

pd1-prepare:
	python scripts/pd1_prepare_lunghist700_binary.py --source data/raw/LungHist700/data/images --output data/lunghist700_binary --summary evidence/public/pd1/dataset_split_summary.json

pd1-train:
	python scripts/pd1_train_swin_lunghist700.py --data-dir data/lunghist700_binary --epochs 1 --batch-size 4 --num-workers 0 --lr 1e-4 --output evidence/public/pd1/pd1_metrics.json
