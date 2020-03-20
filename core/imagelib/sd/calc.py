import numpy as np
import numpy.linalg as npla

def dist_to_edges(pts, p):
    a = pts[:-1,:]
    b = pts[1:,:]
    edges = np.concatenate( ( pts[:-1,None,:], pts[1:,None,:] ), axis=-2)
    
    pa = p-a
    ba = b-a
       
    h = np.clip( np.einsum('ij,ij->i', pa, ba) / np.einsum('ij,ij->i', ba, ba), 0, 1 )

    return npla.norm ( pa - ba*h[...,None], axis=1 )
    
def nearest_edge_id_and_dist(pts, p):
    x = dist_to_edges(pts, p)
    if len(x) != 0:
        return np.argmin(x), np.min(x)
    return None, None