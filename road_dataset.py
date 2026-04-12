import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset


class DeepGlobeRoadDataset(Dataset):
    def __init__(self, root_dir, crop_size=512):
        self.root_dir = root_dir
        self.crop_size = crop_size

        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.endswith("_sat.jpg")
        ][:1000]  # Limit to 1000 for faster training

    def __len__(self):
        return len(self.image_files)

    def random_crop(self, image, mask):
        h, w, _ = image.shape
        ch = self.crop_size
        cw = self.crop_size

        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)

        image = image[top:top+ch, left:left+cw]
        mask = mask[top:top+ch, left:left+cw]

        return image, mask

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(
            self.root_dir,
            img_name.replace("_sat.jpg", "_mask.png")
        )

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        # Convert mask to binary (white = road)
        mask = (mask[:, :, 0] == 255).astype(np.float32)

        # Random crop 512×512
        image, mask = self.random_crop(image, mask)

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask