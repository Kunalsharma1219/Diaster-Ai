import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.terrain_dataset import TerrainDataset
from backend.terrain_model import TerrainUNet

# ======================
# DEVICE
# ======================

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ======================
# CONFIG
# ======================

BATCH_SIZE = 1
IMG_SIZE = 512
EPOCHS = 25
LR = 1e-4

# ======================
# CREATE MODELS DIRECTORY
# ======================

os.makedirs("models", exist_ok=True)

# ======================
# DATASET
# ======================

dataset = TerrainDataset(img_size=IMG_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Total training samples: {len(dataset)}")

# ======================
# MODEL
# ======================

model = TerrainUNet(n_classes=5).to(DEVICE)

# Class weights (background, road, flood, vegetation, building)
class_weights = torch.tensor(
    [1.0, 3.0, 4.0, 1.5, 6.0],
    device=DEVICE
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ======================
# TRAINING
# ======================

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, masks in loop:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    scheduler.step()

    epoch_loss = total_loss / len(loader)
    print(f"\nEpoch {epoch+1} Loss: {epoch_loss:.4f}")

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "models/terrain_model_best.pth")
        print("✅ Best model saved.")

# Save final model
torch.save(model.state_dict(), "models/terrain_model_last.pth")

print("\nTraining Complete.")
print("Best model: models/terrain_model_best.pth")
print("Last model: models/terrain_model_last.pth")