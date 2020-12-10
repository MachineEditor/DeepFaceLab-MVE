from pathlib import Path

class FrameInfo(object):
    def __init__(self, filepath=None, landmarks_list=None, image_to_face_mat=None, aligned_size=512):
        self.filepath = filepath
        self.landmarks_list = landmarks_list or []
        self.motion_deg = 0
        self.motion_power = 0
        self.image_to_face_mat = image_to_face_mat
        self.aligned_size = aligned_size