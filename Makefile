

.PHONY: help install install-dev test test-cov lint format format-check type-check train clean pre-commit all

help:
	@echo "Available commands:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make test             Run tests ith coverage report"
	@echo "  make lint             Run linting (flake8)"
	@echo "  make format           Format code with black and isort"
	@echo "  make train            Train model (use SCRIPT=path/to/script.py)" 
	@echo "  make type-check       Run type checking with mypy"
	@echo "  make clean            Remove build artifacts and cache files"
	@echo "  make all              Run format, lint, type-check, and test"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest --cov --cov-report=html --cov-report=term

lint:
	flake8 models utils tests

format:
	black models utils tests
	isort models utils tests

type-check:
	mypy models utils


clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	rm -rf build dist htmlcov .coverage

all: format lint type-check test
