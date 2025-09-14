import os
import torch
import torchvision.models as models

model_path = "models/eye_disease_model.pth"

# Check if model already exists
if not os.path.exists(model_path):
    print("Model not found. Creating model...")
    model = models.resnet18(pretrained=True)  # Example model
    torch.save(model.state_dict(), model_path)
    print("Model created and saved at:", model_path)
else:
    print("Model already exists. Skipping creation.")
