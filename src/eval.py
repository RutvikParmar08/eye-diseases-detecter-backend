# import tensorflow as tf
# from dataset import load_datasets

# MODEL_PATH = "models/best_model.h5"
# model = tf.keras.models.load_model(MODEL_PATH)

# _, val_ds = load_datasets("data")
# loss, acc, auc = model.evaluate(val_ds)
# print(f"Accuracy: {acc:.3f}, AUC: {auc:.3f}")

import torch
from src.dataset import get_dataloaders
from src.model import EyeDiseaseNet

def evaluate_model():
    data_dir = "data"
    test_loader, classes = get_dataloaders(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = EyeDiseaseNet(num_classes=len(classes))
    model.load_state_dict(torch.load("models/eye_disease_model.pth"))
    model.to(device)
    model.eval()


    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print(f"Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    evaluate_model()