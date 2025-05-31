# crnn-captcha-break

## Description    
A PyTorch-based CAPTCHA recognition system using a CRNN (Convolutional Recurrent Neural Network) architecture. This project combines CNNs for visual feature extraction and LSTMs for sequence modeling to decode fixed-length text CAPTCHAs.

> ⚠️ **Disclaimer**:
> This project is for **research and educational purposes only**. The CAPTCHA images are collected from the [Taiwan Stock Exchange site](https://bsr.twse.com.tw/bshtm/), and we strictly follow its [Terms of Use](https://bsr.twse.com.tw/bshtm/use.htm).
> This repository is a **proof of concept**, and the author takes **no legal responsibility** for any misuse.

---

## Features

* Automatically collects and filters CAPTCHA images
* Uses CRNN architecture (CNN + LSTM)
* Visualizes training history
* Supports GPU training with PyTorch
* Reproducible Python environment using [`uv`](https://github.com/astral-sh/uv)

---

## Project Structure

```
crnn-captcha-break/
├── dataset/                      # Collected CAPTCHA images and labels
├── src/
│   ├── collect_data.py          # CAPTCHA crawler
│   ├── clean_data.py            # Data cleaning and filtering logic
│   ├── crawler_example.py       # CAPTCHA test request example
│   ├── cuda_test.py             # Checks if CUDA is available
│   ├── train_model.py           # Main training script
│   ├── model.py                 # CRNN model definition
│   ├── utils.py                 # Helper functions
│   ├── config.py                # Training hyperparameters and config
│   └── predict.py               # Predict CAPTCHA from image
├── captcha_crnn_best_model.pth  # Best trained model
├── training_history.png         # Training/validation accuracy/loss chart
├── requirements.txt             # Python dependencies
├── uv.lock                      # Lockfile for uv
└── README.md                    # Project documentation
```

---

## Getting Started

### 1. Install Python Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) (a faster pip replacement) for environment and dependency management:

```bash
# Install uv if not yet installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv
uv sync
```

### 2. Check CUDA Compatibility

Make sure your installed PyTorch version matches your CUDA version.
To verify CUDA availability:

```bash
python src/cuda_test.py
```

---

## Usage

#### Collect CAPTCHA Images

```bash
python src/collect_data.py
```

#### Clean CAPTCHA Images Dataset

```bash
python src/clean_data.py
```

#### Train the Model

```bash
python src/train_model.py
```

The script will save the best model as `captcha_crnn_model.pth` and generate a training history plot as `training_history.png`.

## Model Architecture

This project uses a CRNN:

* **CNN layers** extract spatial features from CAPTCHA images.
* **Bidirectional LSTM** layers model the sequential nature of characters.
* **CrossEntropyLoss** is used instead of CTC, since CAPTCHA characters are fixed-length and well-aligned.

## License
This code is for educational use only. Please follow all legal guidelines when using CAPTCHA data.