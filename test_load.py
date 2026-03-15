import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Dataset base path
base_path = "../data/Leviar"

A_path = os.path.join(base_path, "A")
B_path = os.path.join(base_path, "B")

image_name = os.listdir(A_path)[0]

before = cv2.imread(os.path.join(A_path, image_name))
after = cv2.imread(os.path.join(B_path, image_name))

# Convert to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Absolute difference
diff = cv2.absdiff(before_gray, after_gray)

# Threshold to detect change
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Load ground truth label
label_path = os.path.join(base_path, "label", image_name)
label = cv2.imread(label_path, 0)

# Resize if needed
if label.shape != thresh.shape:
    label = cv2.resize(label, (thresh.shape[1], thresh.shape[0]))

# Convert to binary
_, label_bin = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)

# Compute IoU
intersection = np.logical_and(thresh, label_bin)
union = np.logical_or(thresh, label_bin)

iou_score = np.sum(intersection) / np.sum(union)

print("IoU Score:", round(iou_score, 4))

# Clean noise
kernel = np.ones((3,3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Create overlay (red highlight)
overlay = before.copy()
overlay[thresh == 255] = [0, 0, 255]

# Display everything
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.title("Before")
plt.imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2,2,2)
plt.title("After")
plt.imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2,2,3)
plt.title("Change Mask")
plt.imshow(thresh, cmap="gray")
plt.axis("off")

plt.subplot(2,2,4)
plt.title("Change Overlay")
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()