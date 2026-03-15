import torch
import cv2
from src.models.detector import DefectDetector
from fastapi import FastAPI, File, UploadFile
import numpy as np

app = FastAPI(title="Industrial CV Serving API")

# Mock loading
model = DefectDetector(num_classes=3)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocessing
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    
    with torch.no_grad():
        prediction = model([img_tensor])
        
    return {"status": "success", "predictions": str(prediction)}
