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
	python scripts/derive_pspii_weights.py --input reports/pd2/balkrishna_2024_extracted.csv --output reports/pd2/pspii_weights_final.json

pd2-sync-weights:
	python -c "import json, pathlib; p=pathlib.Path('reports/pd2/pspii_weights_final.json'); d=pathlib.Path('data/pspii_weights_final.json'); d.write_text(p.read_text(encoding='utf-8'), encoding='utf-8')"

pd4-benchmark:
	python scripts/integration_benchmark.py --url http://localhost:8000/pspii --calls 100 --polymer PS --output reports/pd4_integration/benchmark_result.json
