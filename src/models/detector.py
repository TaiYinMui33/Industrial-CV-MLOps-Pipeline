import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class DefectDetector(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Load a pre-trained Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the classifier head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    def forward(self, images, targets=None):
        return self.model(images, targets)
