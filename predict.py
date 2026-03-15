import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from model import MiniUNet
from risk_map import generate_risk_map
from path_planner import astar_multi

device = torch.device("cpu")

# Load trained model
model = MiniUNet().to(device)
model.load_state_dict(torch.load("miniunet_model.pth", map_location=device))
model.eval()

# Load sample image
# Load sample image
base_path = "../data/Leviar"
A_path = os.path.join(base_path, "A")
B_path = os.path.join(base_path, "B")

image_list = os.listdir(A_path)[:5]

import random

all_images = os.listdir(A_path)
image_list = random.sample(all_images, min(5, len(all_images)))

for image_name in image_list:

    print("Testing:", image_name)

    before = cv2.imread(os.path.join(A_path, image_name))
    after = cv2.imread(os.path.join(B_path, image_name))

    before_resized = cv2.resize(before, (256, 256)) / 255.0
    after_resized = cv2.resize(after, (256, 256)) / 255.0

    combined = np.concatenate([before_resized, after_resized], axis=2)
    combined = torch.tensor(combined, dtype=torch.float32)\
                    .permute(2, 0, 1)\
                    .unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(combined)
    prediction = torch.sigmoid(output)
    prediction = (prediction > 0.3).float()

prediction = prediction.squeeze().numpy()

# Resize back to original
prediction = cv2.resize(prediction, (before.shape[1], before.shape[0]))

binary_mask = (prediction > 0.5).astype(np.uint8)

# Generate Risk Map
risk_map = generate_risk_map(binary_mask)

# Convert risk map to cost map
cost_map = np.ones_like(risk_map, dtype=np.float32)
cost_map[risk_map == 0] = 1        # safe
cost_map[risk_map == 1] = 3        # moderate
cost_map[risk_map == 2] = 15       # high but not impossible

# Resize for faster A*
small_cost = cv2.resize(cost_map, (128,128), interpolation=cv2.INTER_NEAREST)

start = (10,10)
goal = (110,110)

from path_planner import astar_multi

path = astar_multi(
    small_cost,
    start,
    goal,
    alpha=1.0,
    beta=8.0
)

# Visualize path
path_img = np.zeros((128,128,3), dtype=np.uint8)

if path:
    for p in path:
        path_img[p[0], p[1]] = [0,255,0]

# Overlay for visualization
overlay = before.copy()
overlay[binary_mask == 1] = [0,0,255]

plt.figure(figsize=(14,8))

plt.subplot(2,2,1)
plt.title("Before Image")
plt.imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2,2,2)
plt.title("Predicted Change Mask")
plt.imshow(binary_mask, cmap="gray")
plt.axis("off")

plt.subplot(2,2,3)
plt.title("Risk Map")
plt.imshow(risk_map, cmap="jet")
plt.colorbar()
plt.axis("off")

plt.subplot(2,2,4)
plt.title("Risk-Aware Path")
plt.imshow(path_img)
plt.scatter(start[1], start[0], c='blue', label='Start')
plt.scatter(goal[1], goal[0], c='red', label='Goal')
plt.legend()
plt.axis("off")

plt.tight_layout()
plt.show()