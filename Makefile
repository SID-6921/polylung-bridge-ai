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
	pytest -q

compose-up:
	docker compose up --build
