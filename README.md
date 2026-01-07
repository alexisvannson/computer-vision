# Brand Logo Classification

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)

A PyTorch-based computer vision project for brand logo classification across 7 classes (Apple, Google, Samsung, Facebook, WhatsApp, Messenger, Mozilla). Implements multiple state-of-the-art deep learning architectures with comprehensive training utilities, data augmentation, and hyperparameter tuning capabilities.

## Features

- **Multiple Model Architectures**: CNN, MLP, ResNet18, SENet, Vision Transformer (ViT), and Ensemble methods
- **Advanced Training Pipeline**: Configurable training with data augmentation, learning rate scheduling, and early stopping
- **Class Imbalance Handling**: Weighted loss functions for imbalanced datasets
- **Hyperparameter Tuning**: Automated hyperparameter optimization using Optuna
- **Data Augmentation**: Comprehensive augmentation strategies for improved generalization
- **Kaggle Integration**: Inference utilities for competition submissions
- **CI/CD Integration**: Automated testing and linting with GitHub Actions
- **Code Quality**: Enforced code formatting with Black, import sorting with isort, and type checking with mypy

## Project Structure

```
computer-vision/
├── models/                    # Neural network architectures
│   ├── CNN.py                # Convolutional Neural Network
│   ├── MLP.py                # Multi-layer Perceptron
│   ├── ResNet.py             # ResNet18 implementation
│   ├── SENET.py              # Squeeze-and-Excitation Network
│   ├── VIT.py                # Vision Transformer
│   ├── Ensemble.py           # Ensemble methods for model combinations
│   └── checkpoints/          # Saved model checkpoints
├── utils/                     # Training and utility functions
│   ├── train_model.py        # Main training script
│   ├── tune_hyperparameters.py  # Hyperparameter optimization
│   ├── augment_data.py       # Data augmentation utilities
│   ├── dataloader.py         # Dataset loading and preprocessing
│   └── kaggle_inference.py   # Kaggle submission generation
├── config/                    # Model configuration files (YAML)
│   ├── cnn.yaml
│   ├── mlp.yaml
│   ├── resnet.yaml
│   ├── senet.yaml
│   └── vit.yaml
├── data/                      # Dataset directory
│   ├── train/                # Training images (organized by class)
│   ├── test/                 # Test images for Kaggle submission
│   └── train_labels.csv      # Training labels
├── weights/                   # Pretrained model weights
├── tests/                     # Unit and integration tests
├── notebooks/                 # Jupyter notebooks for experiments
├── .github/workflows/         # CI/CD configuration
│   ├── test.yml              # Automated testing
│   └── lint.yml              # Code quality checks
├── Makefile                   # Development commands
└── TRAINING_GUIDE_95PERCENT.md  # Guide to achieving 95%+ accuracy
```

## Running `train_colab.ipynb` on Google Colab

1. **Open Google Colab**  
   Go to [Google Colab](https://colab.research.google.com/) in your browser.

2. **Load the Notebook**  
   - Click **File > Upload Notebook** if you have `train_colab.ipynb` locally, and select your file.  
   - Alternatively, if the notebook is in a GitHub repo, choose **File > Open notebook > GitHub** and paste the repository link.  
   - You can also drag and drop the notebook into the Colab interface.

3. **Mount Google Drive (optional but recommended)**  
   If your data or weights are stored in Google Drive, insert and run the following cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   This allows you to access files in your Drive.

4. **Check/Set Up Dependencies**  
   - Make sure all necessary packages are installed in the first notebook cells (e.g., `!pip install -r requirements.txt`).
   - You can upload your `requirements.txt` or install packages one by one using pip inside notebook cells, such as:
     ```python
     !pip install torch torchvision optuna
     ```

5. **Upload or Access Data**  
   - If data is in Drive, set data paths appropriately (e.g., `/content/drive/MyDrive/yourfolder/data/`).
   - If you need to upload from your computer, use:
     ```python
     from google.colab import files
     uploaded = files.upload()
     ```

6. **Start Training**  
   - Read through and run the notebook cells in sequence by clicking the "Play" button on the left of each code cell, or select **Runtime > Run all**.
   - Monitor outputs for warnings/errors.
   - Training progress, loss, and accuracy will be shown in the notebook outputs.

7. **Saving Outputs**  
   - After training, save your models or submission files to your Google Drive or download to your machine.

**Tips:**  
- Restart your runtime if you change pip dependencies: **Runtime > Restart runtime**.
- Use a GPU for faster training: **Runtime > Change runtime type > Hardware accelerator > GPU**.

This process will let you easily load and run the `train_colab` notebook on Colab and start training your models end-to-end.
