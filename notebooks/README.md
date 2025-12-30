# Computer Vision - Notebooks
Dear Reader, this readme's goal is for you to run the various notebooks of our
computer-vision project. 
We used the package manager uv.
An important note is that train_colab.ipynb can be run exclusively on Google Colab
since it requires the usage of the various GPU available on Colab.
A second important assumptions is to have the Dataset.zip that you have downloaded on Kaggle in your personnal Drive. It has to be in MyDrive in a folder named "computer-vision-data" and inside of it you must have Dataset.zip. 
So MyDrive --> computer-vision-data --> Dataset.zip. 
All the weights from the trainings are saved in the folder weights.


## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager installed

### Installing uv (if not already installed)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

### 1. Navigate to the project root directory
```bash
cd computer-vision
```

### 2. Install dependencies

```bash
uv pip install -r notebooks/requirements.txt
```

## Available Notebooks

### `eda.ipynb` - Formatted EDA Documentation
**Purpose:** Clean, production-ready EDA with detailed interpretations +
a random forest model to serve as baseline for the project's furture models performance asessement.


### `train_colab.ipynb` - Google Colab Training
**Purpose:** Pipeline for the training of deep learning models on GPU (Google Colab)
