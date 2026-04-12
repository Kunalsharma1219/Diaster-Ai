import os
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import ResNetUNet


# ======================
# CONFIGURATION
# ======================

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/xbd")

IMG_SIZE = 256  # Must match training resolution


# ======================
# LOAD MODEL
# ======================

model = ResNetUNet(num_classes=3).to(DEVICE)

# 🔥 Load Deep Supervision model
model.load_state_dict(
    torch.load("best_resnetunet_deepsup_256.pth", map_location=DEVICE)
)

model.eval()


# ======================
# SELECT RANDOM IMAGE
# ======================

image_dir = os.path.join(DATA_PATH, "images")
mask_dir = os.path.join(DATA_PATH, "targets")

all_images = [
    f for f in os.listdir(image_dir)
    if f.endswith("_pre_disaster.png")
]

image_name = random.choice(all_images)

print("Testing:", image_name)


# ======================
# LOAD IMAGES
# ======================

before = cv2.imread(os.path.join(image_dir, image_name))

post_name = image_name.replace(
    "_pre_disaster.png",
    "_post_disaster.png"
)

after = cv2.imread(os.path.join(image_dir, post_name))

mask_name = image_name.replace(
    "_pre_disaster.png",
    "_post_disaster_target.png"
)

gt_mask = cv2.imread(os.path.join(mask_dir, mask_name), 0)


# ======================
# PREPROCESSING (9 CHANNEL INPUT)
# ======================

before_resized = cv2.resize(before, (IMG_SIZE, IMG_SIZE)) / 255.0
after_resized = cv2.resize(after, (IMG_SIZE, IMG_SIZE)) / 255.0

diff = np.abs(after_resized - before_resized)

combined = np.concatenate(
    [before_resized, after_resized, diff],
    axis=2
)

combined = torch.from_numpy(combined).float()
combined = combined.permute(2, 0, 1).unsqueeze(0)
combined = combined.to(DEVICE)

print("Input shape:", combined.shape)


# ======================
# INFERENCE (Deep Supervision Model)
# ======================

with torch.no_grad():
    main_out, _, _ = model(combined)   # 🔥 unpack outputs
    prediction = torch.argmax(main_out, dim=1)

prediction = prediction.squeeze().cpu().numpy()

# Resize back to original resolution
prediction = cv2.resize(
    prediction,
    (before.shape[1], before.shape[0]),
    interpolation=cv2.INTER_NEAREST
)


# ======================
# OPTIONAL: OVERLAY DAMAGE
# ======================

overlay = before.copy()

# Color damage pixels red
overlay[prediction == 2] = [0, 0, 255]

alpha = 0.4
blended = cv2.addWeighted(overlay, alpha, before, 1 - alpha, 0)


# ======================
# VISUALIZATION
# ======================

plt.figure(figsize=(15, 6))

plt.subplot(1, 4, 1)
plt.title("Before")
plt.imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Ground Truth")
plt.imshow(gt_mask, cmap="jet")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Prediction Mask")
plt.imshow(prediction, cmap="jet")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Overlay")
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()