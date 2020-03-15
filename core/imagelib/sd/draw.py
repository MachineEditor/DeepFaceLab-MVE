"""
Signed distance drawing functions using numpy.
"""

import numpy as np
from numpy import linalg as npla

def circle_faded( hw, center, fade_dists ):
    """
    returns drawn circle in [h,w,1] output range [0..1.0] float32
    
    hw         = [h,w]                      resolution
    center     = [y,x]                      center of circle
    fade_dists = [fade_start, fade_end]     fade values
    """
    h,w = hw      
    
    pts = np.empty( (h,w,2), dtype=np.float32 )
    pts[...,1] = np.arange(h)[None,:]
    pts[...,0] = np.arange(w)[:,None]
    pts = pts.reshape ( (h*w, -1) )

    pts_dists = np.abs ( npla.norm(pts-center, axis=-1) )
    
    if fade_dists[1] == 0:
        fade_dists[1] = 1
        
    pts_dists = ( pts_dists - fade_dists[0] ) / fade_dists[1]
        
    pts_dists = np.clip( 1-pts_dists, 0, 1)
    
    return pts_dists.reshape ( (h,w,1) ).astype(np.float32)
    
def random_circle_faded ( hw, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random
        
    h,w = hw
    hw_max = max(h,w)
    fade_start = rnd_state.randint(hw_max)
    fade_end = fade_start + rnd_state.randint(hw_max- fade_start)
    
    return circle_faded (hw, [ rnd_state.randint(h), rnd_state.randint(w) ], 
                             [fade_start, fade_end] ) 