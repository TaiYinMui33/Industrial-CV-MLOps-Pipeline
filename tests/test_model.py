import torch
import pytest
from src.models.detector import DefectDetector

def test_model_output_shape():
    num_classes = 3
    model = DefectDetector(num_classes=num_classes)
    model.eval()
    
    # Create a dummy image tensor (Batch, Channels, Height, Width)
    dummy_img = [torch.randn(3, 300, 300)]
    
    with torch.no_grad():
        output = model(dummy_img)
        
    # Faster R-CNN output is a list of dicts
    assert isinstance(output, list)
    assert "boxes" in output[0]
    assert "scores" in output[0]
    assert "labels" in output[0]
