import os
import torch
import torchvision.models as models

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model_path = "models/eye_disease_model.pth"
torch.save(model.state_dict(), model_path)

print(f"Model saved at {model_path}")
