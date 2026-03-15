import rasterio
import numpy as np
import cv2

def load_elevation(path):

    with rasterio.open(path) as src:
        elevation = src.read(1)

    return elevation


def compute_slope(elevation):

    gy, gx = np.gradient(elevation)

    slope = np.sqrt(gx**2 + gy**2)

    slope = slope / (slope.max() + 1e-6)

    return slope


def resize_slope_to_image(slope, image_shape):

    h, w = image_shape[:2]

    slope_resized = cv2.resize(slope, (w, h), interpolation=cv2.INTER_LINEAR)

    return slope_resized