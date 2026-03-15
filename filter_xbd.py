import os
import shutil
import json
import numpy as np
import cv2

XBD_PATH = "../data/xbd"
OUTPUT_PATH = "../data/xbd_subset"

images_path = os.path.join(XBD_PATH, "images")
labels_path = os.path.join(XBD_PATH, "labels")
targets_path = os.path.join(XBD_PATH, "targets")

out_before = os.path.join(OUTPUT_PATH, "images_before")
out_after = os.path.join(OUTPUT_PATH, "images_after")
out_masks = os.path.join(OUTPUT_PATH, "masks")

count = 0

for file in os.listdir(labels_path):
    if not file.endswith("_post_disaster.json"):
        continue

    json_path = os.path.join(labels_path, file)

    with open(json_path) as f:
        data = json.load(f)

    base_name = file.replace("_post_disaster.json", "")

# Keep only Hurricane Harvey (Flood)
    if "hurricane-harvey" not in base_name.lower():
        continue

    base_name = file.replace("_post_disaster.json", "")

    pre_img = base_name + "_pre_disaster.png"
    post_img = base_name + "_post_disaster.png"
    target_img = base_name + "_post_disaster_target.png"

    pre_path = os.path.join(images_path, pre_img)
    post_path = os.path.join(images_path, post_img)
    target_path = os.path.join(targets_path, target_img)

    if not os.path.exists(pre_path):
        continue

    # Copy images
    shutil.copy(pre_path, os.path.join(out_before, pre_img))
    shutil.copy(post_path, os.path.join(out_after, post_img))

    # Process mask
    mask = cv2.imread(target_path, 0)

    new_mask = np.zeros_like(mask)

    # Damage classes (2,3,4) -> class 2
    new_mask[(mask == 2) | (mask == 3) | (mask == 4)] = 2

    cv2.imwrite(os.path.join(out_masks, target_img), new_mask)

    count += 1
    print("Processed:", base_name)

print("Total Flood Samples:", count)