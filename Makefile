

# Capture model name argument for train-model, docker-train-model, and tune-model targets
ifneq (,$(filter train-model docker-train-model tune-model,$(firstword $(MAKECMDGOALS))))
  MODEL_ARG := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(MODEL_ARG):;@:)
endif

.PHONY: help install install-dev test test-cov lint format format-check type-check train-model tune-model kaggle-inference clean pre-commit all docker-build docker-up docker-down docker-train-model

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
	@echo "  make tune-model MODEL_NAME [TRIALS=50]  Hyperparameter tuning"
	@echo "                        Example: make tune-model cnn TRIALS=30"
	@echo "  make kaggle-inference MODEL=<model> WEIGHTS=<path> [OUTPUT=submission.csv]"
	@echo "                        Run inference on test set and create submission"
	@echo "                        Example: make kaggle-inference MODEL=resnet WEIGHTS=models/checkpoints/resnet.pth"
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

tune-model:
	@if [ -z "$(MODEL_ARG)" ]; then \
		echo "Error: Model name required"; \
		echo "Usage: make tune-model MODEL_NAME [TRIALS=50]"; \
		echo "Example: make tune-model cnn TRIALS=30"; \
		echo "Available models: senet, vit, mlp, cnn, resnet"; \
		exit 1; \
	fi
	@echo "Tuning hyperparameters for model: $(MODEL_ARG)"
	@echo "Number of trials: $(if $(TRIALS),$(TRIALS),50)"
	python utils/tune_hyperparameters.py --model $(MODEL_ARG) --n-trials $(if $(TRIALS),$(TRIALS),50)

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
