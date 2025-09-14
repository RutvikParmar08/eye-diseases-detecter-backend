
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import EyeDiseaseNet


def train_model():
    data_dir = "data"
    train_loader, classes = get_dataloaders(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = EyeDiseaseNet(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(10):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)


            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}")


    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/eye_disease_model.pth")
    print("Model saved!")


if __name__ == "__main__":
    train_model()