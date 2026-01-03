.PHONY: help install test run docker-build docker-up docker-down docker-logs clean

help:
	@echo "IBKR Trading Bot - Available Commands"
	@echo "======================================"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make run          - Run bot locally (dry run)"
	@echo "  make run-live     - Run bot locally (LIVE MODE)"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Start containers"
	@echo "  make docker-down  - Stop containers"
	@echo "  make docker-logs  - View container logs"
	@echo ""
	@echo "  make clean        - Remove cache files"

install:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

test:
	./venv/bin/python test_connection.py
	./venv/bin/python test_data_layer.py
	./venv/bin/python test_indicators.py
	./venv/bin/python test_telegram.py

run:
	./venv/bin/python -m src.bot --once

run-live:
	@echo "WARNING: This will execute REAL trades!"
	@read -p "Type 'yes' to confirm: " confirm && [ "$$confirm" = "yes" ]
	./venv/bin/python -m src.bot --live --once

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-restart:
	docker compose restart trading-bot

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
