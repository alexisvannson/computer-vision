

# Capture model name argument for train-model and docker-train-model targets
ifneq (,$(filter train-model docker-train-model,$(firstword $(MAKECMDGOALS))))
  MODEL_ARG := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(MODEL_ARG):;@:)
endif

.PHONY: help install install-dev test test-cov lint format format-check type-check train-model kaggle-inference clean pre-commit all docker-build docker-up docker-down docker-train-model

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
	@echo "  make kaggle-inference MODEL=<model> WEIGHTS=<path> [OUTPUT=submission.csv]"
	@echo "                        Run inference on test set and create submission"
	@echo "                        Example: make kaggle-inference MODEL=resnet WEIGHTS=models/checkpoints/resnet.pth"
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

kaggle-inference:
	@if [ -z "$(MODEL)" ] || [ -z "$(WEIGHTS)" ]; then \
		echo "Error: MODEL and WEIGHTS are required"; \
		echo "Usage: make kaggle-inference MODEL=<model> WEIGHTS=<path> [OUTPUT=submission.csv]"; \
		echo "Example: make kaggle-inference MODEL=resnet WEIGHTS=models/checkpoints/resnet.pth"; \
		exit 1; \
	fi
	@echo "Running inference with model: $(MODEL)"
	@echo "Using weights: $(WEIGHTS)"
	python utils/kaggle_inference.py --model $(MODEL) --weights $(WEIGHTS) $(if $(OUTPUT),--output $(OUTPUT),)

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	rm -rf build dist htmlcov .coverage

all: format lint type-check test
