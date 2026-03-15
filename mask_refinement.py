import cv2
import numpy as np


class MaskRefiner:
    def __init__(
        self,
        min_area=100,
        kernel_size=5,
        safety_buffer=3
    ):
        self.min_area = min_area
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.safety_buffer = safety_buffer

    def remove_small_objects(self, mask):

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8
        )

        cleaned = np.zeros_like(mask)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                cleaned[labels == i] = 1

        return cleaned

    def morphological_cleanup(self, mask):
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.kernel)
        return opened

    def add_safety_buffer(self, mask):
        if self.safety_buffer <= 0:
            return mask

        buffer_kernel = np.ones(
            (self.safety_buffer, self.safety_buffer),
            np.uint8
        )

        dilated = cv2.dilate(mask, buffer_kernel, iterations=1)
        return dilated

    def refine(self, prediction_mask):

        # Extract damage class only
        damage_mask = (prediction_mask == 2).astype(np.uint8)

        damage_mask = self.remove_small_objects(damage_mask)
        damage_mask = self.morphological_cleanup(damage_mask)
        damage_mask = self.add_safety_buffer(damage_mask)

        return damage_mask