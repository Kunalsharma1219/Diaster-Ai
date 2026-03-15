import cv2
import numpy as np

def detect_damage(pre, post):

    pre = cv2.resize(pre, (512,512))
    post = cv2.resize(post, (512,512))

    pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(pre_gray, post_gray)

    _, mask = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)

    mask = (mask > 0).astype(np.uint8)

    return mask