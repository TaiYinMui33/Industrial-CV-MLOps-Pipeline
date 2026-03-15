# 🏭 Industrial CV MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) pipeline built for training, tracking, and deploying deep learning models (Object Detection) in industrial defect detection scenarios.

## 🌟 Key Features
- **Data Engineering:** PyTorch custom datasets with robust augmentation strategies using `albumentations`.
- **Model Architecture:** Integrated with YOLO/Detectron2 for high-accuracy bounding box detection.
- **Experiment Tracking:** Native `MLflow` integration for logging metrics, hyperparameters, and model artifacts.
- **Reproducibility:** Packaged as an MLflow Project for seamless execution across different environments.

## 📂 Project Structure
- `src/data/` - Dataloaders and augmentation pipelines.
- `src/models/` - Neural network architectures and wrappers.
- `src/training/` - Training loops and optimization routines.
- `src/evaluation/` - Inference and metric calculation scripts.
- `MLproject` - Environment and entry point definitions for MLflow.

## 🚀 Quick Start
To run the training pipeline with MLflow:
```bash
mlflow run . -P epochs=50 -P batch_size=16
```
