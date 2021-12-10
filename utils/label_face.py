import cv2
from pathlib import Path
import numpy as np
import os

def label_face_filename(face, filename):
    text = os.path.splitext(os.path.basename(filename))[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (5, face.shape[0] - 10)
    thickness = 1
    fontScale = 0.5
    color = (255, 255, 255)
    face = face.copy() # numpy array issue
    cv2.putText(face, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    return face
