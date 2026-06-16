.PHONY: lint test test-all coverage docker-build

lint:
	ruff check app/ tests/

test:
	pytest -m "not llm" --cov=app --cov-report=term-missing

test-all:
	pytest --cov=app --cov-report=term-missing

coverage:
	pytest --cov=app --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

docker-build:
	docker build -t ai-triage-agent:latest .
	docker build -t ai-triage-agent-proxy:latest ./proxy
