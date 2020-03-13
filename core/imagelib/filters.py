import numpy as np
from .blursharpen import LinearMotionBlur
import cv2

def apply_random_hsv_shift(img, rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random
    
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    h = ( h + rnd_state.randint(360) ) % 360
    s = np.clip ( s + rnd_state.random()-0.5, 0, 1 )
    v = np.clip ( v + rnd_state.random()/2-0.25, 0, 1 )                    
    img = np.clip( cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR) , 0, 1 )
    return img
    
def apply_random_motion_blur( img, chance, mb_max_size, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random
    
    mblur_rnd_kernel = rnd_state.randint(mb_max_size)+1
    mblur_rnd_deg    = rnd_state.randint(360)

    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        img = LinearMotionBlur (img, mblur_rnd_kernel, mblur_rnd_deg )
        
    return img
    
def apply_random_gaussian_blur( img, chance, kernel_max_size, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random
        
    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        gblur_rnd_kernel = rnd_state.randint(kernel_max_size)*2+1
        img = cv2.GaussianBlur(img, (gblur_rnd_kernel,)*2 , 0)
    
    return img
    
    
def apply_random_bilinear_resize( img, chance, max_size_per, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random

    if rnd_state.randint(100) < np.clip(chance, 0, 100):
        h,w,c = img.shape
        
        trg = rnd_state.rand()
        rw = w - int( trg * int(w*(max_size_per/100.0)) )                        
        rh = h - int( trg * int(h*(max_size_per/100.0)) )   
             
        img = cv2.resize (img, (rw,rh), cv2.INTER_LINEAR )
        img = cv2.resize (img, (w,h), cv2.INTER_LINEAR )
        
    return img