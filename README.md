# Computer Vision Project

![Tests](https://github.com/alexisvannson/computer-vision/actions/workflows/test.yml/badge.svg)
![Lint](https://github.com/alexisvannson/computer-vision/actions/workflows/lint.yml/badge.svg)

A PyTorch-based computer vision project implementing various neural network architectures including CNNs, MLPs, and exploring advanced topics like attention mechanisms, transfer learning, and generative models.

## Features

- **Flexible CNN Implementation**: Convolutional Neural Network for image classification tasks
- **Customizable MLP**: Multi-layer Perceptron with configurable hidden layers, activations, and normalization
- **Training Utilities**: Modular training pipeline for model development
- **CI/CD Integration**: Automated testing and linting with GitHub Actions
- **Code Quality**: Enforced code formatting with Black, import sorting with isort, and type checking with mypy

## Project Structure

```
computer-vision/
├── models/              # Neural network architectures
│   ├── CNN.py          # Convolutional Neural Network
│   └── MLP.py          # Multi-layer Perceptron
├── utils/              # Training and utility functions
│   └── train_model.py  # Model training utilities
├── tests/              # Unit and integration tests
│   ├── test_models.py
│   └── test_training.py
├── data/               # Dataset directory
│   ├── train/
│   ├── test/
│   └── train_labels.csv
├── notebooks/          # Jupyter notebooks for experiments
├── .github/workflows/  # CI/CD configuration
│   ├── test.yml       # Automated testing
│   └── lint.yml       # Code quality checks
└── Makefile           # Development commands
```

## Installation

### Prerequisites

- Python 3.12+
- pip or uv package manager

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd computer-vision

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install development dependencies (includes testing and linting tools)
make install-dev
# or
pip install -r requirements-dev.txt
```

## Usage


## Development

### Available Make Commands

```bash
# Install development dependencies
make install-dev

# Run tests
make test

# Run tests with coverage report
make test-cov

# Lint code
make lint

# Format code
make format

# Check formatting without changes
make format-check

# Type checking
make type-check

# Train a model
make train SCRIPT=your_script.py

# Clean build artifacts
make clean
```

### Code Quality

This project enforces code quality through:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Static type checking
- **pytest**: Unit and integration testing

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test file
pytest tests/test_models.py
```

## CI/CD

The project includes GitHub Actions workflows for:

- **Automated Testing**: Runs on every push and pull request
- **Code Linting**: Ensures code quality standards
- **Type Checking**: Validates type annotations


### Code Style

- Follow PEP 8 guidelines
- Use type hints for function arguments and return values
- Write docstrings for classes and functions
- Maintain test coverage for new features


