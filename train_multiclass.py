import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from model import ResNetUNet


# ======================
# Configuration
# ======================

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/xbd")

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 35
LR = 3e-4
NUM_CLASSES = 3


# ======================
# Dataset
# ======================

class MultiClassDataset(Dataset):
    def __init__(self, root_dir, clean_ratio=0.6):

        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "targets")

        all_images = os.listdir(self.image_dir)

        damage_images = []
        clean_images = []

        for name in all_images:

            if not name.endswith("_pre_disaster.png"):
                continue

            mask_name = name.replace(
                "_pre_disaster.png",
                "_post_disaster_target.png"
            )

            mask_path = os.path.join(self.mask_dir, mask_name)

            if not os.path.exists(mask_path):
                continue

            mask = cv2.imread(mask_path, 0)
            if mask is None:
                continue

            if np.sum(mask >= 2) > 50:
                damage_images.append(name)
            else:
                clean_images.append(name)

        num_clean_keep = int(clean_ratio * len(clean_images))
        selected_clean = clean_images[:num_clean_keep]

        self.image_names = damage_images + selected_clean
        random.shuffle(self.image_names)

        print(f"Damage images: {len(damage_images)}")
        print(f"Clean images used: {len(selected_clean)}")
        print(f"Total samples: {len(self.image_names)}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img_name = self.image_names[idx]

        before_path = os.path.join(self.image_dir, img_name)
        post_name = img_name.replace("_pre_disaster.png", "_post_disaster.png")
        after_path = os.path.join(self.image_dir, post_name)

        mask_name = img_name.replace("_pre_disaster.png", "_post_disaster_target.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        before = cv2.imread(before_path)
        after = cv2.imread(after_path)
        mask = cv2.imread(mask_path, 0)

        before = cv2.resize(before, (IMG_SIZE, IMG_SIZE)) / 255.0
        after = cv2.resize(after, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_NEAREST)

        mask = np.where(mask >= 2, 2, mask)
        mask = np.where(mask == 1, 0, mask)

        if random.random() > 0.5:
            before = np.fliplr(before).copy()
            after = np.fliplr(after).copy()
            mask = np.fliplr(mask).copy()

        if random.random() > 0.5:
            before = np.flipud(before).copy()
            after = np.flipud(after).copy()
            mask = np.flipud(mask).copy()

        diff = np.abs(after - before)

        combined = np.concatenate([before, after, diff], axis=2)
        combined = torch.from_numpy(combined).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return combined, mask


# ======================
# Loss Functions
# ======================

class FocalLoss(nn.Module):
    def __init__(self, device, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        class_weights = torch.tensor([1.0, 1.0, 4.0]).to(device)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def tversky_loss(pred, target, alpha=0.7, beta=0.3, smooth=1e-6):

    pred = torch.softmax(pred, dim=1)

    pred_damage = pred[:, 2, :, :]
    target_damage = (target == 2).float()

    TP = (pred_damage * target_damage).sum()
    FN = ((1 - pred_damage) * target_damage).sum()
    FP = (pred_damage * (1 - target_damage)).sum()

    tversky = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
    return 1 - tversky


# ======================
# IoU
# ======================

def compute_iou(pred, target, cls):
    pred_cls = (pred == cls)
    target_cls = (target == cls)

    intersection = (pred_cls & target_cls).sum().item()
    union = (pred_cls | target_cls).sum().item()

    return 0.0 if union == 0 else intersection / union


# ======================
# Data Loaders
# ======================

dataset = MultiClassDataset(DATA_PATH)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ======================
# Model
# ======================

model = ResNetUNet(num_classes=3).to(DEVICE)

focal_loss_fn = FocalLoss(device=DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=4
)

best_damage_iou = 0.0


# ======================
# Training Loop
# ======================

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        main_out, ds2_out, ds3_out = model(images)

        f_main = focal_loss_fn(main_out, masks)
        t_main = tversky_loss(main_out, masks)

        f_ds2 = focal_loss_fn(ds2_out, masks)
        t_ds2 = tversky_loss(ds2_out, masks)

        f_ds3 = focal_loss_fn(ds3_out, masks)
        t_ds3 = tversky_loss(ds3_out, masks)

        loss = (
            f_main + 6.0 * t_main +
            0.5 * (f_ds2 + 6.0 * t_ds2) +
            0.3 * (f_ds3 + 6.0 * t_ds3)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    total_iou_bg = 0
    total_iou_damage = 0
    count = 0

    with torch.no_grad():
        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            main_out, _, _ = model(images)
            preds = torch.argmax(main_out, dim=1)

            total_iou_bg += compute_iou(preds, masks, 0)
            total_iou_damage += compute_iou(preds, masks, 2)
            count += 1

    avg_iou_bg = total_iou_bg / count
    avg_iou_damage = total_iou_damage / count

    scheduler.step(avg_iou_damage)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Val Background IoU: {avg_iou_bg:.4f}")
    print(f"Val Damage IoU: {avg_iou_damage:.4f}")

    if avg_iou_damage > best_damage_iou:
        best_damage_iou = avg_iou_damage
        torch.save(model.state_dict(), "resnet34_deepsup_tversky_256_v3.pth")
        print("🔥 Saved BEST model!")

print("\nTraining complete.")