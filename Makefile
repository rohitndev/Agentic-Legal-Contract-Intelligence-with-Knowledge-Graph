.PHONY: install install-cloud demo api test lint docker compose clean

install:
	python -m venv venv && venv/bin/pip install -r requirements.txt

install-cloud:
	venv/bin/pip install -r requirements-cloud.txt

demo:
	python -m scripts.run_demo

api:
	uvicorn api.main:app --reload

test:
	pytest -q

lint:
	ruff check .

docker:
	docker build -t legal-intelligence:latest -f deployment/Dockerfile .

compose:
	docker compose -f deployment/docker-compose.yml up --build

clean:
	rm -rf storage __pycache__ .pytest_cache .ruff_cache
