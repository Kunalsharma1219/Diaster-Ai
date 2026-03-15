import os
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from backend.model import ResNetUNet
from vision.post_processing import MaskRefinement


# ======================
# CONFIGURATION
# ======================

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/xbd")

IMG_SIZE = 256


# ======================
# LOAD MODEL
# ======================

model = ResNetUNet(num_classes=3).to(DEVICE)
model.load_state_dict(
    torch.load("models/best_resnetunet_deepsup_256.pth", map_location=DEVICE)
)
model.eval()

refiner = MaskRefinement()


# ======================
# SELECT IMAGE
# ======================

image_dir = os.path.join(DATA_PATH, "images")

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


# ======================
# PREPROCESS
# ======================

before_resized = cv2.resize(before, (IMG_SIZE, IMG_SIZE)) / 255.0
after_resized = cv2.resize(after, (IMG_SIZE, IMG_SIZE)) / 255.0

diff = np.abs(after_resized - before_resized)

combined = np.concatenate(
    [before_resized, after_resized, diff],
    axis=2
)

combined = torch.from_numpy(combined).float()
combined = combined.permute(2, 0, 1).unsqueeze(0).to(DEVICE)


# ======================
# INFERENCE
# ======================

with torch.no_grad():
    main_out, _, _ = model(combined)
    prediction = torch.argmax(main_out, dim=1)

prediction = prediction.squeeze().cpu().numpy()

prediction = cv2.resize(
    prediction,
    (before.shape[1], before.shape[0]),
    interpolation=cv2.INTER_NEAREST
)


# ======================
# REFINEMENT
# ======================

clean_mask, obstacle_map = refiner.refine(prediction)


# ======================
# VISUALIZATION
# ======================

plt.figure(figsize=(15, 8))

plt.subplot(1, 4, 1)
plt.title("Before")
plt.imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Raw Damage Mask")
plt.imshow(prediction, cmap="jet")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Refined Damage Mask")
plt.imshow(clean_mask, cmap="jet")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Obstacle Map (Inflated)")
plt.imshow(obstacle_map, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()