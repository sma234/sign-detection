import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from dataset import SignDataset
from model import SmallSignNet
from classes import CLASSES

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader = DataLoader(SignDataset("dataset"), batch_size=32, shuffle=True)

    model = SmallSignNet(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(15):
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/15  loss={running_loss:.3f}  acc={correct/total:.3f}")

    torch.save(model.state_dict(), "traffic_signs.pth")
    print("Saved traffic_signs.pth")
