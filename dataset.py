import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class ChangeDetectionDataset(Dataset):
    def __init__(self, base_path):
        self.A_path = os.path.join(base_path, "A")
        self.B_path = os.path.join(base_path, "B")
        self.label_path = os.path.join(base_path, "label")

        self.image_names = os.listdir(self.A_path)[:200]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        before = cv2.imread(os.path.join(self.A_path, image_name))
        after = cv2.imread(os.path.join(self.B_path, image_name))
        label = cv2.imread(os.path.join(self.label_path, image_name), 0)

        before = cv2.resize(before, (256, 256))
        after = cv2.resize(after, (256, 256))
        label = cv2.resize(label, (256, 256))

        before = before / 255.0
        after = after / 255.0
        label = label / 255.0

        # Stack before and after as 6-channel input
        combined = np.concatenate([before, after], axis=2)

        combined = torch.tensor(combined, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return combined, label