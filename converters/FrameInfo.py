class FrameInfo(object):
    def __init__(self, filename=None, landmarks_list=None):
        self.filename = filename
        self.landmarks_list = landmarks_list or []
        self.motion_deg = 0
        self.motion_power = 0