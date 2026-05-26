.PHONY: install install-dev features train inference test lint clean all

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

features:
	python entrypoint/train.py --skip-features || python -c "\
	import sys; sys.path.insert(0,''); \
	from pathlib import Path; from config import CFG; \
	from src.pipelines.feature_eng_pipeline import run; \
	run(Path(CFG['paths']['raw']), Path(CFG['paths']['preprocessed']), Path(CFG['paths']['features']))"

train:
	python entrypoint/train.py

inference:
	python entrypoint/inference.py --all

predict:
	@echo "Usage: make predict HOME='Brazil' AWAY='Argentina' STAGE='final'"
	python entrypoint/inference.py --predict "$(HOME)" "$(AWAY)" --stage $(STAGE)

simulate:
	python entrypoint/inference.py --simulate --n-sims 10000

rankings:
	python entrypoint/inference.py --rankings

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ entrypoint/ config/ tests/
	black --check src/ entrypoint/ config/ tests/

format:
	black src/ entrypoint/ config/ tests/
	ruff check --fix src/ entrypoint/ config/ tests/

docker-build:
	docker compose build

docker-train:
	docker compose run train

docker-inference:
	docker compose run inference

docker-test:
	docker compose run test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -f data/02-preprocessed/*.csv
	rm -f data/03-features/*.csv
	rm -f data/04-predictions/*.pkl
	rm -f data/04-predictions/*.csv

all: install train inference

streamlit:
	streamlit run app/streamlit/app.py --server.port 8501

api:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

dev:
	@echo "Starting API on :8000 and Streamlit on :8501"
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &
	streamlit run app/streamlit/app.py --server.port 8501
