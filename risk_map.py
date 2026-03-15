import cv2
import numpy as np

def generate_risk_map(binary_mask):
    """
    Convert binary mask to 3-level risk map:
    0 = Low Risk
    1 = Medium Risk
    2 = High Risk
    """

    distance = cv2.distanceTransform(
        (binary_mask == 0).astype(np.uint8),
        cv2.DIST_L2,
        5
    )

    risk_map = np.zeros_like(binary_mask)

    # High risk (direct change)
    risk_map[binary_mask == 1] = 2

    # Medium risk (buffer around change)
    risk_map[(distance < 10) & (binary_mask == 0)] = 1

    return risk_map