# deepfake_detection.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load the PyTorch deepfake detection model
model = torch.load("model.pth", map_location=torch.device('cpu'))
model.eval()

# Transformation: adjust if your model expects different input size or normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def detect_deepfake(image_pil):
    """
    Runs deepfake detection on a PIL image.
    Returns:
        - label: 'Real' or 'Fake'
        - confidence: probability score
    """
    image = transform(image_pil).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()
        label = "Fake" if probability >= 0.5 else "Real"
        confidence = round(probability if label == "Fake" else 1 - probability, 2)
        return label, confidence
