import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from backend.road_dataset import DeepGlobeRoadDataset
from backend.road_model import ResNet34UNet

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_PATH = "data/deepglobe/train"
EPOCHS = 15
BATCH_SIZE = 1
LR = 1e-4

# -----------------------------
# Dataset
# -----------------------------
dataset = DeepGlobeRoadDataset(DATA_PATH, crop_size=512)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# -----------------------------
# Model
# -----------------------------
model = ResNet34UNet().to(DEVICE)

# Stronger positive weight for thin roads
pos_weight = torch.tensor([4.0]).to(DEVICE)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def dice_loss(logits, target, smooth=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum()
    union = probs.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

optimizer = optim.Adam(model.parameters(), lr=LR)

# 🔥 LR scheduler (helps when plateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
    verbose=True
)

best_iou = 0
early_stop_counter = 0
early_stop_patience = 6

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        loss = bce(outputs, masks) + dice_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()

        # 🔥 Gradient clipping (stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    iou_total = 0
    count = 0

    with torch.no_grad():
        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            intersection = (preds * masks).sum()
            union = (preds + masks).clamp(0,1).sum()

            iou = (intersection / (union + 1e-6)).item()

            iou_total += iou
            count += 1

    avg_iou = iou_total / count

    scheduler.step(avg_iou)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val IoU: {avg_iou:.4f}")

    if avg_iou > best_iou:
        best_iou = avg_iou
        torch.save(model.state_dict(), "best_road_model_512.pth")
        print("🔥 Saved Best Road Model")
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    # 🔥 Early stopping safeguard
    if early_stop_counter >= early_stop_patience:
        print("⏹ Early stopping triggered.")
        break

print("\nTraining complete.")