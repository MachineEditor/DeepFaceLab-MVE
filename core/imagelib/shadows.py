import numpy as np
import cv2

# https://github.com/OsamaMazhar/Random-Shadows-Highlights
# img is in format np 0-1, float
def shadow_highlights_augmentation(img, seed=None):
    rnd_state = np.random.RandomState (seed)

    high_ratio = (1, 2)
    low_ratio = (0.01, 0.5)
    left_low_ratio = (0.4, 0.6)
    left_high_ratio = (0, 0.2)
    right_low_ratio = (0.4, 0.6)
    right_high_ratio = (0, 0.2)

    # check
    img = np.clip(img*255, 0, 255).astype(np.uint8)

    w, h, _ = img.shape

    high_bright_factor = rnd_state.uniform(high_ratio[0], high_ratio[1])
    low_bright_factor = rnd_state.uniform(low_ratio[0], low_ratio[1])

    left_low_factor = rnd_state.uniform(left_low_ratio[0]*h, left_low_ratio[1]*h)
    left_high_factor = rnd_state.uniform(left_high_ratio[0]*h, left_high_ratio[1]*h)
    right_low_factor = rnd_state.uniform(right_low_ratio[0]*h, right_low_ratio[1]*h)
    right_high_factor = rnd_state.uniform(right_high_ratio[0]*h, right_high_ratio[1]*h)

    tl = (0, left_high_factor)
    bl = (0, left_high_factor+left_low_factor)

    tr = (w, right_high_factor)
    br = (w, right_high_factor+right_low_factor)

    contour = np.array([tl, tr, br, bl], dtype=np.int32)

    mask = np.zeros(img.shape, dtype=img.dtype)
    cv2.fillPoly(mask, [contour], (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    inverted_mask = cv2.bitwise_not(mask)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.multiply(hsv[..., 2], high_bright_factor)
    high_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    hsv[..., 2] = cv2.multiply(hsv[..., 2], low_bright_factor)
    low_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    for i in range(3):
        img[:, :, i] = img[:, :, i] * (mask/255) + high_brightness[:, :, i] * (1-mask/255)
        img[:, :, i] = img[:, :, i] * (inverted_mask/255) + low_brightness[:, :, i] * (1-inverted_mask/255)

    img = np.clip(img/255.0, 0, 1).astype(np.float32)

    return img
