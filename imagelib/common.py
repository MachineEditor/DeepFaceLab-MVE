import numpy as np

def normalize_channels(img, target_channels):
    img_shape_len = len(img.shape)
    if img_shape_len == 2:
        h, w = img.shape
        c = 0
    elif img_shape_len == 3:
        h, w, c = img.shape
    else:
        raise ValueError("normalize: incorrect image dimensions.")

    if c == 0 and target_channels > 0:
        img = img[...,np.newaxis]
        c = 1
        
    if c == 1 and target_channels > 1:
        img = np.repeat (img, target_channels, -1)
        c = target_channels
        
    if c > target_channels:
        img = img[...,0:target_channels]
        c = target_channels

    return img

def overlay_alpha_image(img_target, img_source, xy_offset=(0,0) ):
    (h,w,c) = img_source.shape
    if c != 4:
        raise ValueError("overlay_alpha_image, img_source must have 4 channels")

    x1, x2 = xy_offset[0], xy_offset[0] + w
    y1, y2 = xy_offset[1], xy_offset[1] + h

    alpha_s = img_source[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        img_target[y1:y2, x1:x2, c] = (alpha_s * img_source[:, :, c] +
                                        alpha_l * img_target[y1:y2, x1:x2, c])