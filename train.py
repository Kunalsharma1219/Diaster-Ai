import torch
from torch.utils.data import DataLoader
from dataset import ChangeDetectionDataset
from model import MiniUNet
import torch.nn as nn
import torch.optim as optim

device = torch.device("cpu")

dataset = ChangeDetectionDataset("../data/Leviar")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = MiniUNet().to(device)

# --- BCE + Dice combined loss ---
bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1.
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss_bce = bce(outputs, labels)
        loss_dice = dice_loss(outputs, labels)
        loss = loss_bce + loss_dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# --------- Evaluation ---------
def calculate_iou(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.3).float()

    intersection = (pred * target).sum()
    union = ((pred + target) > 0).float().sum()

    return (intersection / union).item()

model.eval()

with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        iou = calculate_iou(outputs, labels)
        print("Model IoU:", round(iou, 4))
        break

torch.save(model.state_dict(), "miniunet_model.pth")
print("Model saved successfully.")