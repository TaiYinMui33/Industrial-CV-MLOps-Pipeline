# 🏭 Industrial CV MLOps Pipeline

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview
**Industrial CV MLOps Pipeline** is a specialized framework designed to bridge the gap between deep learning research and industrial production. Focused on **Defect Detection** (cracks, corrosion, anomalies), this repository provides a standardized, reproducible pipeline for training, tracking, and serving Computer Vision models.

This project implements the core tenets of MLOps: automation, versioning, and continuous monitoring, specifically tailored for high-accuracy industrial safety requirements.

---

## 🚀 Key MLOps Features
- **Deterministic Data Pipelines:** Integrated with `albumentations` for advanced spatial and pixel-level augmentations, ensuring model robustness in varied lighting and environments.
- **Experiment Tracking (MLflow):** Every training run is automatically logged with its specific hyperparameters, loss curves, and artifact versions.
- **Reproducible Environments:** Uses `conda.yaml` and `MLproject` definitions for consistent execution across cloud and on-premise GPU clusters.
- **Model Registry Capability:** Designed to facilitate easy promotion of models from "Staging" to "Production".
- **Scalable Serving:** Built-in FastAPI server for real-time inference, optimized for high-throughput visual inspection.

---

## 🏗️ Technical Architecture
The pipeline is structured around a decoupled architecture to allow independent scaling of components:

1.  **Data Layer:** Handles raw image ingestion, YOLO-format label parsing, and on-the-fly augmentation.
2.  **Modeling Layer:** Provides a flexible wrapper around `torchvision` and custom backbones (Faster R-CNN, Mask R-CNN).
3.  **Training Engine:** Orchestrates the optimization loop, gradient scaling (Mixed Precision), and validation steps.
4.  **Logging Wrapper:** A unified interface for `MLflow` to track metrics without cluttering core logic.
5.  **Deployment Layer:** Converts trained weights into an optimized inference service.

---

## 📁 Repository Structure
```text
├── configs/            # YAML definitions for models and training
├── scripts/            # Utility scripts (Model serving, Exporting)
├── src/                # Core source code
│   ├── data/           # Dataset loaders and transformers
│   ├── models/         # Architecture definitions
│   ├── training/       # Training logic and MLflow integration
│   └── evaluation/     # Metrics (mAP, Confusion Matrix)
├── tests/              # PyTest suite for data and model logic
├── MLproject           # MLflow project entry points
├── conda.yaml          # Environment specification
└── README.md
```

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (Recommended)
- MLflow installed locally

### Installation & Setup
1. **Clone the repo:**
   ```bash
   git clone https://github.com/TaiYinMui33/Industrial-CV-MLOps-Pipeline.git
   cd Industrial-CV-MLOps-Pipeline
   ```

2. **Setup environment:**
   ```bash
   conda env create -f conda.yaml
   conda activate cv-mlops-env
   ```

### Running the Pipeline
You can run the entire pipeline via MLflow CLI to ensure reproducibility:
```bash
mlflow run . -P epochs=100 -P batch_size=16 -P learning_rate=1e-4
```

To run manually:
```bash
python src/training/train.py --epochs 50 --lr 0.001
```

---

## 📊 Monitoring & Evaluation
Start the MLflow UI to compare experiments:
```bash
mlflow ui --port 5000
```
Then visit `http://localhost:5000` to visualize metrics and download artifacts.

---

## 🔌 Model Serving (Inference)
Launch the production-ready inference API:
```bash
uvicorn scripts.serve_model:app --host 0.0.0.0 --port 8000
```
**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@crack_sample.jpg"
```

---

## 🧪 Quality Assurance
We use `pytest` to validate both the neural network output shapes and the data loading integrity.
```bash
pytest tests/
```

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---
**Developed with ❤️ by [Tai Yin Mui](https://github.com/TaiYinMui33)**  
*Bridging AI Research and Industrial Excellence.*
