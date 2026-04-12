import os
import cv2
import numpy as np
import random
from torch.utils.data import Dataset

class TerrainDataset(Dataset):
    """
    Unified dataset combining:
    - DeepGlobe (roads)
    - Leviar (flood)
    - xbd_subset (damage)
    """

    def __init__(self, root_dir="data", img_size=512, samples_per_dataset=600):
        self.root = root_dir
        self.img_size = img_size

        # ------------------------
        # DeepGlobe (Road)
        # ------------------------
        dg_path = os.path.join(root_dir, "deepglobe", "train")
        self.deepglobe_pairs = []

        for file in os.listdir(dg_path):
            if file.endswith("_sat.jpg"):
                prefix = file.replace("_sat.jpg", "")
                img_path = os.path.join(dg_path, f"{prefix}_sat.jpg")
                mask_path = os.path.join(dg_path, f"{prefix}_mask.png")
                if os.path.exists(mask_path):
                    self.deepglobe_pairs.append(("road", img_path, mask_path))

        # ------------------------
        # Leviar (Flood)
        # ------------------------
        lev_img_dir = os.path.join(root_dir, "Leviar", "A")
        lev_mask_dir = os.path.join(root_dir, "Leviar", "label")

        self.leviar_pairs = []
        for file in os.listdir(lev_img_dir):
            if file.endswith(".png"):
                img_path = os.path.join(lev_img_dir, file)
                mask_path = os.path.join(lev_mask_dir, file)
                if os.path.exists(mask_path):
                    self.leviar_pairs.append(("flood", img_path, mask_path))

        # ------------------------
        # xBD Subset (Damage)
        # ------------------------
        xbd_img_dir = os.path.join(root_dir, "xbd_subset", "images_before")
        xbd_mask_dir = os.path.join(root_dir, "xbd_subset", "masks")

        self.xbd_pairs = []
        for file in os.listdir(xbd_img_dir):
            if file.endswith(".png"):
                img_path = os.path.join(xbd_img_dir, file)
                mask_name = file.replace("_pre_disaster.png",
                                         "_post_disaster_target.png")
                mask_path = os.path.join(xbd_mask_dir, mask_name)
                if os.path.exists(mask_path):
                    self.xbd_pairs.append(("damage", img_path, mask_path))

        # Balanced sampling
        self.samples = []
        self.samples += random.sample(self.deepglobe_pairs,
                                      min(samples_per_dataset, len(self.deepglobe_pairs)))
        self.samples += random.sample(self.leviar_pairs,
                                      min(samples_per_dataset, len(self.leviar_pairs)))
        self.samples += random.sample(self.xbd_pairs,
                                      min(samples_per_dataset + 200, len(self.xbd_pairs)))

        random.shuffle(self.samples)

        print(f"Total training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dataset_type, img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        mask = self.convert_mask(mask, dataset_type)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = mask.astype(np.int64)

        return image, mask

    def convert_mask(self, mask, dataset_type):

        new_mask = np.zeros_like(mask)

        if dataset_type == "road":
            # DeepGlobe roads (white)
            new_mask[mask == 255] = 1  # road
            new_mask[mask == 0] = 0    # background

        elif dataset_type == "flood":
            # Leviar flood (255)
            new_mask[mask == 255] = 2  # flood
            new_mask[mask == 0] = 0    # background

        elif dataset_type == "damage":
            # xBD heavy damage
            new_mask[(mask == 2) | (mask == 3)] = 4  # building/damage
            new_mask[(mask == 0) | (mask == 1)] = 0  # background

        return new_mask