import numpy as np
from .blursharpen import LinearMotionBlur
import cv2

def apply_random_rgb_levels(img, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random
    np_rnd = rnd_state.rand
    
    inBlack  = np.array([np_rnd()*0.25    , np_rnd()*0.25    , np_rnd()*0.25], dtype=np.float32)
    inWhite  = np.array([1.0-np_rnd()*0.25, 1.0-np_rnd()*0.25, 1.0-np_rnd()*0.25], dtype=np.float32)
    inGamma  = np.array([0.5+np_rnd(), 0.5+np_rnd(), 0.5+np_rnd()], dtype=np.float32)
    
    outBlack  = np.array([np_rnd()*0.25    , np_rnd()*0.25    , np_rnd()*0.25], dtype=np.float32)
    outWhite  = np.array([1.0-np_rnd()*0.25, 1.0-np_rnd()*0.25, 1.0-np_rnd()*0.25], dtype=np.float32)

    result = np.clip( (img - inBlack) / (inWhite - inBlack), 0, 1 )
    result = ( result ** (1/inGamma) ) *  (outWhite - outBlack) + outBlack
    result = np.clip(result, 0, 1)
    
    if mask is not None:
        result = img*(1-mask) + result*mask
        
    return result
    
def apply_random_hsv_shift(img, mask=None, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random
    
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    h = ( h + rnd_state.randint(360) ) % 360
    s = np.clip ( s + rnd_state.random()-0.5, 0, 1 )
    v = np.clip ( v + rnd_state.random()-0.5, 0, 1 )                    
    
    result = np.clip( cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR) , 0, 1 )
    if mask is not None:
        result = img*(1-mask) + result*mask
        
    return result
    
def apply_random_motion_blur( img, chance, mb_max_size, mask=None, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random
    
    mblur_rnd_kernel = rnd_state.randint(mb_max_size)+1
    mblur_rnd_deg    = rnd_state.randint(360)

    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        result = LinearMotionBlur (result, mblur_rnd_kernel, mblur_rnd_deg )
        if mask is not None:
            result = img*(1-mask) + result*mask
        
    return result
    
def apply_random_gaussian_blur( img, chance, kernel_max_size, mask=None, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random
        
    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        gblur_rnd_kernel = rnd_state.randint(kernel_max_size)*2+1
        result = cv2.GaussianBlur(result, (gblur_rnd_kernel,)*2 , 0)
        if mask is not None:
            result = img*(1-mask) + result*mask
            
    return result
    
    
def apply_random_bilinear_resize( img, chance, max_size_per, mask=None, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random

    result = img
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        h,w,c = result.shape
        
        trg = rnd_state.rand()
        rw = w - int( trg * int(w*(max_size_per/100.0)) )                        
        rh = h - int( trg * int(h*(max_size_per/100.0)) )   
             
        result = cv2.resize (result, (rw,rh), interpolation=cv2.INTER_LINEAR )
        result = cv2.resize (result, (w,h), interpolation=cv2.INTER_LINEAR )
        if mask is not None:
            result = img*(1-mask) + result*mask
            
    return result