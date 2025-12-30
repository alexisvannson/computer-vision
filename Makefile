

# Capture model name argument for train-model and docker-train-model targets
ifneq (,$(filter train-model docker-train-model,$(firstword $(MAKECMDGOALS))))
  MODEL_ARG := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(MODEL_ARG):;@:)
endif

.PHONY: help install install-dev test test-cov lint format format-check type-check train-model clean pre-commit all docker-build docker-up docker-down docker-train-model

help:
	@echo "Available commands:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make test             Run tests with coverage report"
	@echo "  make lint             Run linting (flake8)"
	@echo "  make format           Format code with black and isort"
	@echo "  make train-model MODEL_NAME  Train a specific model"
	@echo "                        Example: make train-model mlp"
	@echo "                        Available: senet, vit, mlp, cnn, resnet"
	@echo "  make type-check       Run type checking with mypy"
	@echo "  make clean            Remove build artifacts and cache files"
	@echo "  make all              Run format, lint, type-check, and test"
	@echo ""
	@echo "Docker commands:"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-up        Start Docker container"
	@echo "  make docker-down      Stop Docker container"
	@echo "  make docker-train-model MODEL_NAME  Train model in Docker"
	@echo "                        Example: make docker-train-model mlp"

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

train-model:
	@if [ -z "$(MODEL_ARG)" ]; then \
		echo "Error: Model name required"; \
		echo "Usage: make train-model MODEL_NAME"; \
		echo "Example: make train-model mlp"; \
		echo "Available models: senet, vit, mlp, cnn, resnet"; \
		exit 1; \
	fi
	@echo "Training model: $(MODEL_ARG)"
	python utils/train_model.py --model $(MODEL_ARG)

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	rm -rf build dist htmlcov .coverage

all: format lint type-check test

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-train-model:
	@if [ -z "$(MODEL_ARG)" ]; then \
		echo "Error: Model name required"; \
		echo "Usage: make docker-train-model MODEL_NAME"; \
		echo "Example: make docker-train-model mlp"; \
		echo "Available models: senet, vit, mlp, cnn, resnet"; \
		exit 1; \
	fi
	@echo "Training model in Docker: $(MODEL_ARG)"
	docker-compose run --rm computer-vision python -m utils.train_model --model $(MODEL_ARG)
