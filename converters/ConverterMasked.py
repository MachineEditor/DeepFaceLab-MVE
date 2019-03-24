import traceback
from .Converter import Converter
from facelib import LandmarksProcessor
from facelib import FaceType
from facelib import FANSegmentator
import cv2
import numpy as np
from utils import image_utils
from interact import interact as io
'''
default_mode = {1:'overlay',
             2:'hist-match',
             3:'hist-match-bw',
             4:'seamless',
             5:'seamless-hist-match',
             6:'raw'}
'''
class ConverterMasked(Converter):

    #override
    def __init__(self,  predictor_func,
                        predictor_input_size=0,
                        output_size=0,
                        face_type=FaceType.FULL,
                        default_mode = 4,
                        base_erode_mask_modifier = 0,
                        base_blur_mask_modifier = 0,
                        default_erode_mask_modifier = 0,
                        default_blur_mask_modifier = 0,
                        clip_hborder_mask_per = 0):

        super().__init__(predictor_func, Converter.TYPE_FACE)
        self.predictor_input_size = predictor_input_size
        self.output_size = output_size
        self.face_type = face_type
        self.clip_hborder_mask_per = clip_hborder_mask_per

        mode = io.input_int ("Choose mode: (1) overlay, (2) hist match, (3) hist match bw, (4) seamless, (5) raw. Default - %d : " % (default_mode) , default_mode)

        mode_dict = {1:'overlay',
                     2:'hist-match',
                     3:'hist-match-bw',
                     4:'seamless',
                     5:'raw'}

        self.mode = mode_dict.get (mode, mode_dict[default_mode] )
        self.suppress_seamless_jitter = False

        if self.mode == 'raw':
            mode = io.input_int ("Choose raw mode: (1) rgb, (2) rgb+mask (default), (3) mask only, (4) predicted only : ", 2)
            self.raw_mode = {1:'rgb',
                             2:'rgb-mask',
                             3:'mask-only',
                             4:'predicted-only'}.get (mode, 'rgb-mask')

        if self.mode != 'raw':

            if self.mode == 'seamless':
                io.input_bool ("Suppress seamless jitter? [ y/n ] (?:help skip:n ) : ", False, help_message="Seamless clone produces face jitter. You can suppress it, but process can take a long time." )

                if io.input_bool("Seamless hist match? (y/n skip:n) : ", False):
                    self.mode = 'seamless-hist-match'

            if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
                self.masked_hist_match = io.input_bool("Masked hist match? (y/n skip:y) : ", True)

            if self.mode == 'hist-match' or self.mode == 'hist-match-bw' or self.mode == 'seamless-hist-match':
                self.hist_match_threshold = np.clip ( io.input_int("Hist match threshold [0..255] (skip:255) :  ", 255), 0, 255)

        if face_type == FaceType.FULL:
            self.mask_mode = io.input_int ("Mask mode: (1) learned, (2) dst, (3) FAN-prd, (4) FAN-dst (?) help. Default - %d : " % (1) , 1, help_message="If you learned mask, then option 1 should be choosed. 'dst' mask is raw shaky mask from dst aligned images. 'FAN-prd' - using super smooth mask by pretrained FAN-model from predicted face. 'FAN-dst' - using super smooth mask by pretrained FAN-model from dst face.")
        else:
            self.mask_mode = io.input_int ("Mask mode: (1) learned, (2) dst . Default - %d : " % (1) , 1)

        if self.mask_mode == 3 or self.mask_mode == 4:
            self.fan_seg = None

        if self.mode != 'raw':
            self.erode_mask_modifier = base_erode_mask_modifier + np.clip ( io.input_int ("Choose erode mask modifier [-200..200] (skip:%d) : " % (default_erode_mask_modifier), default_erode_mask_modifier), -200, 200)
            self.blur_mask_modifier = base_blur_mask_modifier + np.clip ( io.input_int ("Choose blur mask modifier [-200..200] (skip:%d) : " % (default_blur_mask_modifier), default_blur_mask_modifier), -200, 200)

            self.seamless_erode_mask_modifier = 0
            if 'seamless' in self.mode:
                self.seamless_erode_mask_modifier = np.clip ( io.input_int ("Choose seamless erode mask modifier [-100..100] (skip:0) : ", 0), -100, 100)

        self.output_face_scale = np.clip ( 1.0 + io.input_int ("Choose output face scale modifier [-50..50] (skip:0) : ", 0)*0.01, 0.5, 1.5)
        self.color_transfer_mode = io.input_str ("Apply color transfer to predicted face? Choose mode ( rct/lct skip:None ) : ", None, ['rct','lct'])

        if self.mode != 'raw':
            self.final_image_color_degrade_power = np.clip (  io.input_int ("Degrade color power of final image [0..100] (skip:0) : ", 0), 0, 100)
            self.alpha = io.input_bool("Export png with alpha channel? (y/n skip:n) : ", False)

        io.log_info ("")
        self.over_res = 4 if self.suppress_seamless_jitter else 1

    #override
    def dummy_predict(self):
        self.predictor_func ( np.zeros ( (self.predictor_input_size,self.predictor_input_size,4), dtype=np.float32 ) )

    #overridable
    def on_cli_initialize(self):
        if (self.mask_mode == 3 or self.mask_mode == 4) and self.fan_seg == None:
            self.fan_seg = FANSegmentator(256, FaceType.toString(FaceType.FULL) )

    #override
    def convert_face (self, img_bgr, img_face_landmarks, debug):
        if self.over_res != 1:
            img_bgr = cv2.resize ( img_bgr, ( img_bgr.shape[1]*self.over_res, img_bgr.shape[0]*self.over_res ) )
            img_face_landmarks = img_face_landmarks*self.over_res

        if debug:
            debugs = [img_bgr.copy()]

        img_size = img_bgr.shape[1], img_bgr.shape[0]

        img_face_mask_a = LandmarksProcessor.get_image_hull_mask (img_bgr.shape, img_face_landmarks)

        face_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, self.output_size, face_type=self.face_type)
        face_output_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, self.output_size, face_type=self.face_type, scale=self.output_face_scale)

        dst_face_bgr      = cv2.warpAffine( img_bgr        , face_mat, (self.output_size, self.output_size), flags=cv2.INTER_LANCZOS4 )
        dst_face_mask_a_0 = cv2.warpAffine( img_face_mask_a, face_mat, (self.output_size, self.output_size), flags=cv2.INTER_LANCZOS4 )

        predictor_input_bgr      = cv2.resize (dst_face_bgr,      (self.predictor_input_size,self.predictor_input_size))
        predictor_input_mask_a_0 = cv2.resize (dst_face_mask_a_0, (self.predictor_input_size,self.predictor_input_size))
        predictor_input_mask_a   = np.expand_dims (predictor_input_mask_a_0, -1)

        predicted_bgra = self.predictor_func ( np.concatenate( (predictor_input_bgr, predictor_input_mask_a), -1) )

        prd_face_bgr      = np.clip (predicted_bgra[:,:,0:3], 0, 1.0 )
        prd_face_mask_a_0 = np.clip (predicted_bgra[:,:,3], 0.0, 1.0)

        if self.mask_mode == 2: #dst
            prd_face_mask_a_0 = predictor_input_mask_a_0
        elif self.mask_mode == 3: #FAN-prd
            prd_face_bgr_256 = cv2.resize (prd_face_bgr, (256,256) )
            prd_face_bgr_256_mask = self.fan_seg.extract_from_bgr( np.expand_dims(prd_face_bgr_256,0) ) [0]
            prd_face_mask_a_0 = cv2.resize (prd_face_bgr_256_mask, (self.predictor_input_size, self.predictor_input_size))
        elif self.mask_mode == 4: #FAN-dst
            face_256_mat     = LandmarksProcessor.get_transform_mat (img_face_landmarks, 256, face_type=FaceType.FULL)
            dst_face_256_bgr = cv2.warpAffine(img_bgr, face_256_mat, (256, 256), flags=cv2.INTER_LANCZOS4 )
            dst_face_256_mask = self.fan_seg.extract_from_bgr( np.expand_dims(dst_face_256_bgr,0) ) [0]
            prd_face_mask_a_0 = cv2.resize (dst_face_256_mask, (self.predictor_input_size, self.predictor_input_size))

        prd_face_mask_a_0[ prd_face_mask_a_0 < 0.001 ] = 0.0

        prd_face_mask_a   = np.expand_dims (prd_face_mask_a_0, axis=-1)
        prd_face_mask_aaa = np.repeat (prd_face_mask_a, (3,), axis=-1)

        img_face_mask_aaa = cv2.warpAffine( prd_face_mask_aaa, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4 )
        img_face_mask_aaa = np.clip (img_face_mask_aaa, 0.0, 1.0)
        img_face_mask_aaa [ img_face_mask_aaa <= 0.1 ] = 0.0 #get rid of noise

        if debug:
            debugs += [img_face_mask_aaa.copy()]

        if 'seamless' in self.mode:
            #mask used for cv2.seamlessClone
            img_face_seamless_mask_aaa = None
            for i in range(9, 0, -1):
                a = img_face_mask_aaa > i / 10.0
                if len(np.argwhere(a)) == 0:
                    continue
                img_face_seamless_mask_aaa = img_face_mask_aaa.copy()
                img_face_seamless_mask_aaa[a] = 1.0
                img_face_seamless_mask_aaa[img_face_seamless_mask_aaa <= i / 10.0] = 0.0

        out_img = img_bgr.copy()

        if self.mode == 'raw':
            if self.raw_mode == 'rgb' or self.raw_mode == 'rgb-mask':
                out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, out_img, cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )

            if self.raw_mode == 'rgb-mask':
                out_img = np.concatenate ( [out_img, np.expand_dims (img_face_mask_aaa[:,:,0],-1)], -1 )

            if self.raw_mode == 'mask-only':
                out_img = img_face_mask_aaa

            if self.raw_mode == 'predicted-only':
                out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(out_img.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )

        elif ('seamless' not in self.mode) or (img_face_seamless_mask_aaa is not None):
            #averaging [lenx, leny, maskx, masky] by grayscale gradients of upscaled mask
            ar = []
            for i in range(1, 10):
                maxregion = np.argwhere( img_face_mask_aaa > i / 10.0 )
                if maxregion.size != 0:
                    miny,minx = maxregion.min(axis=0)[:2]
                    maxy,maxx = maxregion.max(axis=0)[:2]
                    lenx = maxx - minx
                    leny = maxy - miny
                    maskx = ( minx+(lenx/2) )
                    masky = ( miny+(leny/2) )
                    if lenx >= 4 and leny >= 4:
                        ar += [ [ lenx, leny, maskx, masky]  ]

            if len(ar) > 0:
                lenx, leny, maskx, masky = np.mean ( ar, axis=0 )

                if debug:
                    io.log_info ("lenx/leny:(%d/%d) maskx/masky:(%f/%f)" % (lenx, leny, maskx, masky  ) )

                maskx = int( maskx )
                masky = int( masky )

                lowest_len = min (lenx, leny)

                if debug:
                    io.log_info ("lowest_len = %f" % (lowest_len) )

                img_mask_blurry_aaa = img_face_mask_aaa

                if self.erode_mask_modifier != 0:
                    ero  = int( lowest_len * ( 0.126 - lowest_len * 0.00004551365 ) * 0.01*self.erode_mask_modifier )
                    if debug:
                        io.log_info ("erode_size = %d" % (ero) )
                    if ero > 0:
                        img_mask_blurry_aaa = cv2.erode(img_mask_blurry_aaa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )
                    elif ero < 0:
                        img_mask_blurry_aaa = cv2.dilate(img_mask_blurry_aaa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(-ero,-ero)), iterations = 1 )

                if self.seamless_erode_mask_modifier != 0:
                    ero  = int( lowest_len * ( 0.126 - lowest_len * 0.00004551365 ) * 0.01*self.seamless_erode_mask_modifier )
                    if debug:
                        io.log_info ("seamless_erode_size = %d" % (ero) )
                    if ero > 0:
                        img_face_seamless_mask_aaa = cv2.erode(img_face_seamless_mask_aaa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )
                    elif ero < 0:
                        img_face_seamless_mask_aaa = cv2.dilate(img_face_seamless_mask_aaa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(-ero,-ero)), iterations = 1 )
                    img_face_seamless_mask_aaa = np.clip (img_face_seamless_mask_aaa, 0, 1)

                if self.clip_hborder_mask_per > 0: #clip hborder before blur
                    prd_hborder_rect_mask_a = np.ones ( prd_face_mask_a.shape, dtype=np.float32)
                    prd_border_size = int ( prd_hborder_rect_mask_a.shape[1] * self.clip_hborder_mask_per )
                    prd_hborder_rect_mask_a[:,0:prd_border_size,:] = 0
                    prd_hborder_rect_mask_a[:,-prd_border_size:,:] = 0
                    prd_hborder_rect_mask_a = np.expand_dims(cv2.blur(prd_hborder_rect_mask_a, (prd_border_size, prd_border_size) ),-1)

                    img_prd_hborder_rect_mask_a = cv2.warpAffine( prd_hborder_rect_mask_a, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4 )
                    img_prd_hborder_rect_mask_a = np.expand_dims (img_prd_hborder_rect_mask_a, -1)
                    img_mask_blurry_aaa *= img_prd_hborder_rect_mask_a
                    img_mask_blurry_aaa = np.clip( img_mask_blurry_aaa, 0, 1.0 )

                    if debug:
                        debugs += [img_mask_blurry_aaa.copy()]

                if self.blur_mask_modifier > 0:
                    blur = int( lowest_len * 0.10 * 0.01*self.blur_mask_modifier )
                    if debug:
                        io.log_info ("blur_size = %d" % (blur) )
                    if blur > 0:
                        img_mask_blurry_aaa = cv2.blur(img_mask_blurry_aaa, (blur, blur) )

                img_mask_blurry_aaa = np.clip( img_mask_blurry_aaa, 0, 1.0 )

                if debug:
                    debugs += [img_mask_blurry_aaa.copy()]

                if self.color_transfer_mode is not None:
                    if self.color_transfer_mode == 'rct':
                        if debug:
                            debugs += [ np.clip( cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]

                        prd_face_bgr = image_utils.reinhard_color_transfer ( np.clip( (prd_face_bgr*255).astype(np.uint8), 0, 255),
                                                                             np.clip( (dst_face_bgr*255).astype(np.uint8), 0, 255),
                                                                             source_mask=prd_face_mask_a, target_mask=prd_face_mask_a)
                        prd_face_bgr = np.clip( prd_face_bgr.astype(np.float32) / 255.0, 0.0, 1.0)

                        if debug:
                            debugs += [ np.clip( cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]


                    elif self.color_transfer_mode == 'lct':
                        if debug:
                            debugs += [ np.clip( cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]

                        prd_face_bgr = image_utils.linear_color_transfer (prd_face_bgr, dst_face_bgr)
                        prd_face_bgr = np.clip( prd_face_bgr, 0.0, 1.0)

                        if debug:
                            debugs += [ np.clip( cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]

                if self.mode == 'hist-match-bw':
                    prd_face_bgr = cv2.cvtColor(prd_face_bgr, cv2.COLOR_BGR2GRAY)
                    prd_face_bgr = np.repeat( np.expand_dims (prd_face_bgr, -1), (3,), -1 )

                if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
                    if debug:
                        debugs += [ cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ) ]

                    hist_mask_a = np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)

                    if self.masked_hist_match:
                        hist_mask_a *= prd_face_mask_a

                    hist_match_1 = prd_face_bgr*hist_mask_a + (1.0-hist_mask_a)* np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)
                    hist_match_1[ hist_match_1 > 1.0 ] = 1.0

                    hist_match_2 = dst_face_bgr*hist_mask_a + (1.0-hist_mask_a)* np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)
                    hist_match_2[ hist_match_1 > 1.0 ] = 1.0

                    prd_face_bgr = image_utils.color_hist_match(hist_match_1, hist_match_2, self.hist_match_threshold )

                if self.mode == 'hist-match-bw':
                    prd_face_bgr = prd_face_bgr.astype(dtype=np.float32)

                out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, out_img, cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )
                out_img = np.clip(out_img, 0.0, 1.0)

                if debug:
                    debugs += [out_img.copy()]

                if self.mode == 'overlay':
                    pass

                if 'seamless' in self.mode:
                    try:
                        out_img = cv2.seamlessClone( (out_img*255).astype(np.uint8), (img_bgr*255).astype(np.uint8), (img_face_seamless_mask_aaa*255).astype(np.uint8), (maskx,masky) , cv2.NORMAL_CLONE )
                        out_img = out_img.astype(dtype=np.float32) / 255.0
                    except Exception as e:
                        #seamlessClone may fail in some cases
                        e_str = traceback.format_exc()

                        if 'MemoryError' in e_str:
                            raise Exception("Seamless fail: " + e_str) #reraise MemoryError in order to reprocess this data by other processes
                        else:
                            print ("Seamless fail: " + e_str)

                    if debug:
                        debugs += [out_img.copy()]

                out_img = np.clip( img_bgr*(1-img_mask_blurry_aaa) + (out_img*img_mask_blurry_aaa) , 0, 1.0 )

                if self.mode == 'seamless-hist-match':
                    out_face_bgr = cv2.warpAffine( out_img, face_mat, (self.output_size, self.output_size) )
                    new_out_face_bgr = image_utils.color_hist_match(out_face_bgr, dst_face_bgr, self.hist_match_threshold)
                    new_out = cv2.warpAffine( new_out_face_bgr, face_mat, img_size, img_bgr.copy(), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )
                    out_img =  np.clip( img_bgr*(1-img_mask_blurry_aaa) + (new_out*img_mask_blurry_aaa) , 0, 1.0 )

                if self.final_image_color_degrade_power != 0:
                    if debug:
                        debugs += [out_img.copy()]
                    out_img_reduced = image_utils.reduce_colors(out_img, 256)
                    if self.final_image_color_degrade_power == 100:
                        out_img = out_img_reduced
                    else:
                        alpha = self.final_image_color_degrade_power / 100.0
                        out_img = (out_img*(1.0-alpha) + out_img_reduced*alpha)

                if self.alpha:
                    out_img = np.concatenate ( [out_img, np.expand_dims (img_mask_blurry_aaa[:,:,0],-1)], -1 )

        if self.over_res != 1:
            out_img = cv2.resize ( out_img, ( img_bgr.shape[1] // self.over_res, img_bgr.shape[0] // self.over_res ) )

        out_img = np.clip (out_img, 0.0, 1.0 )

        if debug:
            debugs += [out_img.copy()]

        return debugs if debug else out_img

