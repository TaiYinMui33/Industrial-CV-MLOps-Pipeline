import argparse
import mlflow
import torch
from src.models.detector import DefectDetector

def train(epochs: int, batch_size: int, lr: float):
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Defect_Detection_v1")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr
        })
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DefectDetector(num_classes=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        print(f"Starting training for {epochs} epochs on {device}...")
        
        # Mock training loop
        for epoch in range(epochs):
            # train_loss = train_one_epoch(...)
            train_loss = 0.5 / (epoch + 1)
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}")
            
        # Log final model
        mlflow.pytorch.log_model(model, "model")
        print("Training complete and model logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train(args.epochs, args.batch_size, args.lr)
