import numpy as np
import copy

from facelib import FaceType
from interact import interact as io


class ConverterConfig(object):
    TYPE_NONE = 0
    TYPE_MASKED = 1
    TYPE_FACE_AVATAR = 2
    ####

    TYPE_IMAGE = 3
    TYPE_IMAGE_WITH_LANDMARKS = 4

    def __init__(self, type=0):
        self.type = type

        self.superres_func = None
        self.blursharpen_func = None
        self.fanseg_input_size = None
        self.fanseg_extract_func = None
        self.ebs_ct_func = None

        self.super_res_dict = {0:"None", 1:'RankSRGAN'}
        self.sharpen_dict = {0:"None", 1:'box', 2:'gaussian'}

        #default changeable params
        self.super_resolution_mode = 0
        self.sharpen_mode = 0
        self.blursharpen_amount = 0

    def copy(self):
        return copy.copy(self)

    #overridable
    def ask_settings(self):
        s = """Choose sharpen mode: \n"""
        for key in self.sharpen_dict.keys():
            s += f"""({key}) {self.sharpen_dict[key]}\n"""
        s += f"""?:help Default: {list(self.sharpen_dict.keys())[0]} : """
        self.sharpen_mode = io.input_int (s, 0, valid_list=self.sharpen_dict.keys(), help_message="Enhance details by applying sharpen filter.")

        if self.sharpen_mode != 0:
            self.blursharpen_amount = np.clip ( io.input_int ("Choose blur/sharpen amount [-100..100] (skip:0) : ", 0), -100, 100 )

        s = """Choose super resolution mode: \n"""
        for key in self.super_res_dict.keys():
            s += f"""({key}) {self.super_res_dict[key]}\n"""
        s += f"""?:help Default: {list(self.super_res_dict.keys())[0]} : """
        self.super_resolution_mode = io.input_int (s, 0, valid_list=self.super_res_dict.keys(), help_message="Enhance details by applying superresolution network.")

    def toggle_sharpen_mode(self):
        a = list( self.sharpen_dict.keys() )
        self.sharpen_mode = a[ (a.index(self.sharpen_mode)+1) % len(a) ]

    def add_blursharpen_amount(self, diff):
        self.blursharpen_amount = np.clip ( self.blursharpen_amount+diff, -100, 100)

    def toggle_super_resolution_mode(self):
        a = list( self.super_res_dict.keys() )
        self.super_resolution_mode = a[ (a.index(self.super_resolution_mode)+1) % len(a) ]

    #overridable
    def __eq__(self, other):
        #check equality of changeable params

        if isinstance(other, ConverterConfig):
            return self.sharpen_mode == other.sharpen_mode and \
                   self.blursharpen_amount == other.blursharpen_amount and \
                   self.super_resolution_mode == other.super_resolution_mode

        return False

    #overridable
    def to_string(self, filename):
        r = ""
        r += f"sharpen_mode : {self.sharpen_dict[self.sharpen_mode]}\n"
        r += f"blursharpen_amount : {self.blursharpen_amount}\n"        
        r += f"super_resolution_mode : {self.super_res_dict[self.super_resolution_mode]}\n"
        return r
        
mode_dict = {0:'original',
             1:'overlay',
             2:'hist-match',
             3:'seamless2',
             4:'seamless',
             5:'seamless-hist-match',             
             6:'raw-rgb',
             7:'raw-rgb-mask',
             8:'raw-mask-only',
             9:'raw-predicted-only'}

full_face_mask_mode_dict = {1:'learned',
                                    2:'dst',
                                    3:'FAN-prd',
                                    4:'FAN-dst',
                                    5:'FAN-prd*FAN-dst',
                                    6:'learned*FAN-prd*FAN-dst'}

half_face_mask_mode_dict = {1:'learned',
                                    2:'dst',
                                    4:'FAN-dst',
                                    7:'learned*FAN-dst'}

ctm_dict = { 0: "None", 1:"rct", 2:"lct", 3:"mkl", 4:"mkl-m", 5:"idt", 6:"idt-m", 7:"ebs" }
ctm_str_dict = {None:0, "rct":1, "lct":2, "mkl":3, "mkl-m":4, "idt":5, "idt-m":6, "ebs":7 }

class ConverterConfigMasked(ConverterConfig):

    def __init__(self, face_type=FaceType.FULL,
                       default_mode = 4,
                       clip_hborder_mask_per = 0,
                       ):

        super().__init__(type=ConverterConfig.TYPE_MASKED)
        
        self.face_type = face_type
        if self.face_type not in [FaceType.FULL, FaceType.HALF]:
            raise ValueError("ConverterConfigMasked supports only full or half face masks.")

        self.default_mode = default_mode
        self.clip_hborder_mask_per = clip_hborder_mask_per

        #default changeable params
        self.mode = 'overlay'
        self.masked_hist_match = True
        self.hist_match_threshold = 238
        self.mask_mode = 1
        self.erode_mask_modifier = 0
        self.blur_mask_modifier = 0
        self.motion_blur_power = 0
        self.output_face_scale = 0
        self.color_transfer_mode = 0
        self.color_degrade_power = 0
        self.export_mask_alpha = False

    def copy(self):
        return copy.copy(self)

    def set_mode (self, mode):
        self.mode = mode_dict.get (mode, mode_dict[self.default_mode] )

    def toggle_masked_hist_match(self):
        if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
            self.masked_hist_match = not self.masked_hist_match

    def add_hist_match_threshold(self, diff):
        if self.mode == 'hist-match' or self.mode == 'hist-match-bw' or self.mode == 'seamless-hist-match':
            self.hist_match_threshold = np.clip ( self.hist_match_threshold+diff , 0, 255)

    def toggle_mask_mode(self):
        if self.face_type == FaceType.FULL:
            a = list( full_face_mask_mode_dict.keys() )
        else:
            a = list( half_face_mask_mode_dict.keys() )
        self.mask_mode = a[ (a.index(self.mask_mode)+1) % len(a) ]

    def add_erode_mask_modifier(self, diff):
        self.erode_mask_modifier = np.clip ( self.erode_mask_modifier+diff , -400, 400)

    def add_blur_mask_modifier(self, diff):
        self.blur_mask_modifier = np.clip ( self.blur_mask_modifier+diff , -400, 400)

    def add_motion_blur_power(self, diff):
        self.motion_blur_power = np.clip ( self.motion_blur_power+diff, 0, 100)

    def add_output_face_scale(self, diff):
        self.output_face_scale = np.clip ( self.output_face_scale+diff , -50, 50)

    def toggle_color_transfer_mode(self):
        self.color_transfer_mode = (self.color_transfer_mode+1) % ( max(ctm_dict.keys())+1 )

    def add_color_degrade_power(self, diff):
        self.color_degrade_power = np.clip ( self.color_degrade_power+diff , 0, 100)

    def toggle_export_mask_alpha(self):
        self.export_mask_alpha = not self.export_mask_alpha

    def ask_settings(self):

        s = """Choose mode: \n"""
        for key in mode_dict.keys():
            s += f"""({key}) {mode_dict[key]}\n"""
        s += f"""Default: {self.default_mode} : """

        mode = io.input_int (s, self.default_mode)

        self.mode = mode_dict.get (mode, mode_dict[self.default_mode] )

        if 'raw' not in self.mode:
            if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
                self.masked_hist_match = io.input_bool("Masked hist match? (y/n skip:y) : ", True)

            if self.mode == 'hist-match' or self.mode == 'hist-match-bw' or self.mode == 'seamless-hist-match':
                self.hist_match_threshold = np.clip ( io.input_int("Hist match threshold [0..255] (skip:255) :  ", 255), 0, 255)

        if self.face_type == FaceType.FULL:
            s = """Choose mask mode: \n"""
            for key in full_face_mask_mode_dict.keys():
                s += f"""({key}) {full_face_mask_mode_dict[key]}\n"""
            s += f"""?:help Default: 1 : """

            self.mask_mode = io.input_int (s, 1, valid_list=full_face_mask_mode_dict.keys(), help_message="If you learned the mask, then option 1 should be choosed. 'dst' mask is raw shaky mask from dst aligned images. 'FAN-prd' - using super smooth mask by pretrained FAN-model from predicted face. 'FAN-dst' - using super smooth mask by pretrained FAN-model from dst face. 'FAN-prd*FAN-dst' or 'learned*FAN-prd*FAN-dst' - using multiplied masks.")
        else:
            s = """Choose mask mode: \n"""
            for key in half_face_mask_mode_dict.keys():
                s += f"""({key}) {half_face_mask_mode_dict[key]}\n"""
            s += f"""?:help , Default: 1 : """
            self.mask_mode = io.input_int (s, 1, valid_list=half_face_mask_mode_dict.keys(), help_message="If you learned the mask, then option 1 should be choosed. 'dst' mask is raw shaky mask from dst aligned images.")

        if 'raw' not in self.mode:
            self.erode_mask_modifier = np.clip ( io.input_int ("Choose erode mask modifier [-400..400] (skip:%d) : " % 0, 0), -400, 400)
            self.blur_mask_modifier =  np.clip ( io.input_int ("Choose blur mask modifier [-400..400] (skip:%d) : " % 0, 0), -400, 400)
            self.motion_blur_power = np.clip ( io.input_int ("Choose motion blur power [0..100] (skip:%d) : " % (0), 0), 0, 100)

        self.output_face_scale = np.clip (io.input_int ("Choose output face scale modifier [-50..50] (skip:0) : ", 0), -50, 50)

        if 'raw' not in self.mode:
            self.color_transfer_mode = io.input_str ("Apply color transfer to predicted face? Choose mode ( rct/lct/ebs skip:None ) : ", None, ctm_str_dict.keys() )
            self.color_transfer_mode = ctm_str_dict[self.color_transfer_mode]

        super().ask_settings()

        if 'raw' not in self.mode:
            self.color_degrade_power = np.clip (  io.input_int ("Degrade color power of final image [0..100] (skip:0) : ", 0), 0, 100)
            self.export_mask_alpha = io.input_bool("Export png with alpha channel of the mask? (y/n skip:n) : ", False)

        io.log_info ("")

    def __eq__(self, other):
        #check equality of changeable params

        if isinstance(other, ConverterConfigMasked):
            return super().__eq__(other) and \
                   self.mode == other.mode and \
                   self.masked_hist_match == other.masked_hist_match and \
                   self.hist_match_threshold == other.hist_match_threshold and \
                   self.mask_mode == other.mask_mode and \
                   self.erode_mask_modifier == other.erode_mask_modifier and \
                   self.blur_mask_modifier == other.blur_mask_modifier and \
                   self.motion_blur_power == other.motion_blur_power and \
                   self.output_face_scale == other.output_face_scale and \
                   self.color_transfer_mode == other.color_transfer_mode and \
                   self.color_degrade_power == other.color_degrade_power and \
                   self.export_mask_alpha == other.export_mask_alpha

        return False

    def to_string(self, filename):
        r = (
            f"""ConverterConfig {filename}:\n"""
            f"""Mode: {self.mode}\n"""
            )

        if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
            r += f"""masked_hist_match: {self.masked_hist_match}\n"""

        if self.mode == 'hist-match' or self.mode == 'hist-match-bw' or self.mode == 'seamless-hist-match':
            r += f"""hist_match_threshold: {self.hist_match_threshold}\n"""

        if self.face_type == FaceType.FULL:
            r += f"""mask_mode: { full_face_mask_mode_dict[self.mask_mode] }\n"""
        else:
            r += f"""mask_mode: { half_face_mask_mode_dict[self.mask_mode] }\n"""

        if 'raw' not in self.mode:
            r += (f"""erode_mask_modifier: {self.erode_mask_modifier}\n"""
                  f"""blur_mask_modifier: {self.blur_mask_modifier}\n"""
                  f"""motion_blur_power: {self.motion_blur_power}\n""")

        r += f"""output_face_scale: {self.output_face_scale}\n"""

        if 'raw' not in self.mode:
            r += f"""color_transfer_mode: { ctm_dict[self.color_transfer_mode]}\n"""

        r += super().to_string(filename)

        if 'raw' not in self.mode:
            r += (f"""color_degrade_power: {self.color_degrade_power}\n"""
                  f"""export_mask_alpha: {self.export_mask_alpha}\n""")

        r += "================"

        return r


class ConverterConfigFaceAvatar(ConverterConfig):

    def __init__(self, temporal_face_count=0):
        super().__init__(type=ConverterConfig.TYPE_FACE_AVATAR)
        self.temporal_face_count = temporal_face_count

        #changeable params
        self.add_source_image = False

    def copy(self):
        return copy.copy(self)

    #override
    def ask_settings(self):
        self.add_source_image = io.input_bool("Add source image? (y/n ?:help skip:n) : ", False, help_message="Add source image for comparison.")
        super().ask_settings()

    def toggle_add_source_image(self):
        self.add_source_image = not self.add_source_image

    #override
    def __eq__(self, other):
        #check equality of changeable params

        if isinstance(other, ConverterConfigFaceAvatar):
            return super().__eq__(other) and \
                   self.add_source_image == other.add_source_image

        return False

    #override
    def to_string(self, filename):
        return (f"ConverterConfig {filename}:\n"
                f"add_source_image : {self.add_source_image}\n") + \
                super().to_string(filename) + "================"

