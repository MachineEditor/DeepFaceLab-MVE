from models import ConverterBase
from facelib import LandmarksProcessor
from facelib import FaceType
import cv2
import numpy as np
from utils import image_utils
from utils.console_utils import *

class ConverterMasked(ConverterBase):

    #override
    def __init__(self,  predictor,
                        predictor_input_size=0, 
                        output_size=0, 
                        face_type=FaceType.FULL,
                        base_erode_mask_modifier = 0,
                        base_blur_mask_modifier = 0,
                        
                        **in_options):
                        
        super().__init__(predictor)
        self.predictor_input_size = predictor_input_size
        self.output_size = output_size
        self.face_type = face_type 
        self.TFLabConverter = None
        
        mode = input_int ("Choose mode: (1) overlay, (2) hist match, (3) hist match bw, (4) seamless (default), (5) seamless hist match, (6) raw : ", 4)
        self.mode = {1:'overlay',
                     2:'hist-match',
                     3:'hist-match-bw',
                     4:'seamless',
                     5:'seamless-hist-match',
                     6:'raw'}.get (mode, 'seamless')
        
        if self.mode == 'raw':
            mode = input_int ("Choose raw mode: (1) rgb, (2) rgb+mask (default), (3) mask only, (4) predicted only : ", 2)
            self.raw_mode = {1:'rgb',
                             2:'rgb-mask',
                             3:'mask-only',
                             4:'predicted-only'}.get (mode, 'rgb-mask')
        
        if self.mode != 'raw':
            if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
                self.masked_hist_match = input_bool("Masked hist match? (y/n skip:y) : ", True)
            
            if self.mode == 'hist-match' or self.mode == 'hist-match-bw' or self.mode == 'seamless-hist-match':
                self.hist_match_threshold = np.clip ( input_int("Hist match threshold [0..255] (skip:255) :  ", 255), 0, 255)
            
        self.use_predicted_mask = input_bool("Use predicted mask? (y/n skip:y) : ", True)
            
        if self.mode != 'raw':
            self.erode_mask_modifier = base_erode_mask_modifier + np.clip ( input_int ("Choose erode mask modifier [-200..200] (skip:0) : ", 0), -200, 200)
            self.blur_mask_modifier = base_blur_mask_modifier + np.clip ( input_int ("Choose blur mask modifier [-200..200] (skip:0) : ", 0), -200, 200)
            
            self.seamless_erode_mask_modifier = 0
            if self.mode == 'seamless' or self.mode == 'seamless-hist-match':
                self.seamless_erode_mask_modifier = np.clip ( input_int ("Choose seamless erode mask modifier [-100..100] (skip:0) : ", 0), -100, 100)
          
        self.output_face_scale = np.clip ( 1.0 + input_int ("Choose output face scale modifier [-50..50] (skip:0) : ", 0)*0.01, 0.5, 1.5)
        
        if self.mode != 'raw':
            self.transfercolor = input_bool("Transfer color from dst face to converted final face? (y/n skip:n) : ", False)
            self.final_image_color_degrade_power = np.clip (  input_int ("Degrade color power of final image [0..100] (skip:0) : ", 0), 0, 100)
            self.alpha = input_bool("Export png with alpha channel? (y/n skip:n) : ", False)
            
        print ("")
  
    #override
    def get_mode(self):
        return ConverterBase.MODE_FACE
        
    #override
    def dummy_predict(self):
        self.predictor ( np.zeros ( (self.predictor_input_size,self.predictor_input_size,4), dtype=np.float32 ) )
        
    #override
    def convert_face (self, img_bgr, img_face_landmarks, debug):        
        if debug:        
            debugs = [img_bgr.copy()]

        img_size = img_bgr.shape[1], img_bgr.shape[0]

        img_face_mask_a = LandmarksProcessor.get_image_hull_mask (img_bgr, img_face_landmarks)
        
        face_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, self.output_size, face_type=self.face_type)
        face_output_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, self.output_size, face_type=self.face_type, scale=self.output_face_scale)
        
        dst_face_bgr      = cv2.warpAffine( img_bgr        , face_mat, (self.output_size, self.output_size), flags=cv2.INTER_LANCZOS4 )
        dst_face_mask_a_0 = cv2.warpAffine( img_face_mask_a, face_mat, (self.output_size, self.output_size), flags=cv2.INTER_LANCZOS4 )

        predictor_input_bgr      = cv2.resize (dst_face_bgr,      (self.predictor_input_size,self.predictor_input_size))
        predictor_input_mask_a_0 = cv2.resize (dst_face_mask_a_0, (self.predictor_input_size,self.predictor_input_size))
        predictor_input_mask_a   = np.expand_dims (predictor_input_mask_a_0, -1) 
        
        predicted_bgra = self.predictor ( np.concatenate( (predictor_input_bgr, predictor_input_mask_a), -1) )

        prd_face_bgr      = np.clip (predicted_bgra[:,:,0:3], 0, 1.0 )
        prd_face_mask_a_0 = np.clip (predicted_bgra[:,:,3], 0.0, 1.0)
        
        if not self.use_predicted_mask:
            prd_face_mask_a_0 = predictor_input_mask_a_0
            
        prd_face_mask_a_0[ prd_face_mask_a_0 < 0.001 ] = 0.0
        
        prd_face_mask_a   = np.expand_dims (prd_face_mask_a_0, axis=-1)
        prd_face_mask_aaa = np.repeat (prd_face_mask_a, (3,), axis=-1)

        img_prd_face_mask_aaa = cv2.warpAffine( prd_face_mask_aaa, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=float), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4 )
        img_prd_face_mask_aaa = np.clip (img_prd_face_mask_aaa, 0.0, 1.0)
            
        img_face_mask_aaa = img_prd_face_mask_aaa
        
        if debug:
            debugs += [img_face_mask_aaa.copy()]
        
        img_face_mask_aaa [ img_face_mask_aaa <= 0.1 ] = 0.0
            
        img_face_mask_flatten_aaa = img_face_mask_aaa.copy()
        img_face_mask_flatten_aaa[img_face_mask_flatten_aaa > 0.9] = 1.0

        maxregion = np.argwhere(img_face_mask_flatten_aaa==1.0)        

        out_img = img_bgr.copy()
        
        if self.mode == 'raw':
            if self.raw_mode == 'rgb' or self.raw_mode == 'rgb-mask':
                out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, out_img, cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )
                
            if self.raw_mode == 'rgb-mask':
                out_img = np.concatenate ( [out_img, np.expand_dims (img_face_mask_aaa[:,:,0],-1)], -1 )
                
            if self.raw_mode == 'mask-only':
                out_img = img_face_mask_aaa   

            if self.raw_mode == 'predicted-only':
                out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(out_img.shape), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )
                
        else:
            if maxregion.size != 0:
                miny,minx = maxregion.min(axis=0)[:2]
                maxy,maxx = maxregion.max(axis=0)[:2]
                
                if debug:
                    print ("maxregion.size: %d, minx:%d, maxx:%d miny:%d, maxy:%d" % (maxregion.size, minx, maxx, miny, maxy  ) )
                
                lenx = maxx - minx
                leny = maxy - miny
                if lenx >= 4 and leny >= 4:
                    masky = int(minx+(lenx//2))
                    maskx = int(miny+(leny//2))
                    lowest_len = min (lenx, leny)
                    
                    if debug:
                        print ("lowest_len = %f" % (lowest_len) )
                  
                    img_mask_blurry_aaa = img_face_mask_aaa
                    if self.erode_mask_modifier != 0:
                        ero  = int( lowest_len * ( 0.126 - lowest_len * 0.00004551365 ) * 0.01*self.erode_mask_modifier )
                        if debug:
                            print ("erode_size = %d" % (ero) )                    
                        if ero > 0:
                            img_mask_blurry_aaa = cv2.erode(img_mask_blurry_aaa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )                        
                        elif ero < 0:
                            img_mask_blurry_aaa = cv2.dilate(img_mask_blurry_aaa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(-ero,-ero)), iterations = 1 )
                    
                    if self.seamless_erode_mask_modifier != 0:
                        ero  = int( lowest_len * ( 0.126 - lowest_len * 0.00004551365 ) * 0.01*self.seamless_erode_mask_modifier )
                        if debug:
                            print ("seamless_erode_size = %d" % (ero) )
                        if ero > 0:
                            img_face_mask_flatten_aaa = cv2.erode(img_face_mask_flatten_aaa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )
                        elif ero < 0:
                            img_face_mask_flatten_aaa = cv2.dilate(img_face_mask_flatten_aaa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(-ero,-ero)), iterations = 1 )
                        
                            
                    if self.blur_mask_modifier > 0:
                        blur = int( lowest_len * 0.10 * 0.01*self.blur_mask_modifier )
                        if debug:
                            print ("blur_size = %d" % (blur) )
                        if blur > 0:
                            img_mask_blurry_aaa = cv2.blur(img_mask_blurry_aaa, (blur, blur) )                    
                        
                    img_mask_blurry_aaa = np.clip( img_mask_blurry_aaa, 0, 1.0 )

                    if self.mode == 'hist-match-bw':
                        prd_face_bgr = cv2.cvtColor(prd_face_bgr, cv2.COLOR_BGR2GRAY)
                        prd_face_bgr = np.repeat( np.expand_dims (prd_face_bgr, -1), (3,), -1 )
                    
                    if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
                        if debug:
                            debugs += [ cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ) ]
                            
                        hist_mask_a = np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=prd_face_bgr.dtype)
                            
                        if self.masked_hist_match:
                            hist_mask_a *= prd_face_mask_a
                        
                        hist_match_1 = prd_face_bgr*hist_mask_a + (1.0-hist_mask_a)* np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=prd_face_bgr.dtype) 
                        hist_match_1[ hist_match_1 > 1.0 ] = 1.0
                        
                        hist_match_2 = dst_face_bgr*hist_mask_a + (1.0-hist_mask_a)* np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=prd_face_bgr.dtype) 
                        hist_match_2[ hist_match_1 > 1.0 ] = 1.0

                        prd_face_bgr = image_utils.color_hist_match(hist_match_1, hist_match_2, self.hist_match_threshold )
                            
                    if self.mode == 'hist-match-bw':
                        prd_face_bgr = prd_face_bgr.astype(np.float32)

                    out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, out_img, cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )
            
                    if debug:
                        debugs += [out_img.copy()]
                        debugs += [img_mask_blurry_aaa.copy()]

                    if self.mode == 'overlay':
                        pass
                        
                    if self.mode == 'seamless' or self.mode == 'seamless-hist-match':
                        out_img = np.clip( img_bgr*(1-img_face_mask_aaa) + (out_img*img_face_mask_aaa) , 0, 1.0 )
                        if debug:
                            debugs += [out_img.copy()]

                        out_img = cv2.seamlessClone( (out_img*255).astype(np.uint8), (img_bgr*255).astype(np.uint8), (img_face_mask_flatten_aaa*255).astype(np.uint8), (masky,maskx) , cv2.NORMAL_CLONE )
                        out_img = out_img.astype(np.float32) / 255.0
                        
                        if debug:
                            debugs += [out_img.copy()]

                    out_img =  np.clip( img_bgr*(1-img_mask_blurry_aaa) + (out_img*img_mask_blurry_aaa) , 0, 1.0 )

                    if self.mode == 'seamless-hist-match':
                        out_face_bgr = cv2.warpAffine( out_img, face_mat, (self.output_size, self.output_size) )      
                        new_out_face_bgr = image_utils.color_hist_match(out_face_bgr, dst_face_bgr, self.hist_match_threshold)                
                        new_out = cv2.warpAffine( new_out_face_bgr, face_mat, img_size, img_bgr.copy(), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )
                        out_img =  np.clip( img_bgr*(1-img_mask_blurry_aaa) + (new_out*img_mask_blurry_aaa) , 0, 1.0 )
                        
                    if self.transfercolor:  
                        if self.TFLabConverter is None:
                            self.TFLabConverter = image_utils.TFLabConverter() 
                            
                        img_lab_l, img_lab_a, img_lab_b = np.split ( self.TFLabConverter.bgr2lab (img_bgr), 3, axis=-1 )
                        out_img_lab_l, out_img_lab_a, out_img_lab_b = np.split ( self.TFLabConverter.bgr2lab (out_img), 3, axis=-1 )      
                        
                        out_img = self.TFLabConverter.lab2bgr ( np.concatenate([out_img_lab_l, img_lab_a, img_lab_b], axis=-1) )
         
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
       
        if debug:
            debugs += [out_img.copy()]
            
        return debugs if debug else out_img     

        