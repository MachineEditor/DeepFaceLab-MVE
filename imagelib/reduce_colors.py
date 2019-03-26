import numpy as np
import cv2
from PIL import Image

#n_colors = [0..256]
def reduce_colors (img_bgr, n_colors):
    img_rgb = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) * 255.0).astype(np.uint8)
    img_rgb_pil = Image.fromarray(img_rgb)
    img_rgb_pil_p = img_rgb_pil.convert('P', palette=Image.ADAPTIVE, colors=n_colors)

    img_rgb_p = img_rgb_pil_p.convert('RGB')
    img_bgr = cv2.cvtColor( np.array(img_rgb_p, dtype=np.float32) / 255.0, cv2.COLOR_RGB2BGR )

    return img_bgr
