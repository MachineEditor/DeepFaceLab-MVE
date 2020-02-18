import numpy as np
import cv2
from core import randomex

def gen_warp_params (source, flip, rotation_range=[-10,10], scale_range=[-0.5, 0.5], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05], rnd_seed=None  ):
    h,w,c = source.shape
    if (h != w):
        raise ValueError ('gen_warp_params accepts only square images.')

    if rnd_seed != None:
        rnd_state = np.random.RandomState (rnd_seed)
    else:
        rnd_state = np.random

    rotation = rnd_state.uniform( rotation_range[0], rotation_range[1] )
    scale = rnd_state.uniform(1 +scale_range[0], 1 +scale_range[1])
    tx = rnd_state.uniform( tx_range[0], tx_range[1] )
    ty = rnd_state.uniform( ty_range[0], ty_range[1] )
    p_flip = flip and rnd_state.randint(10) < 4

    #random warp by grid
    cell_size = [ w // (2**i) for i in range(1,4) ] [ rnd_state.randint(3) ]
    cell_count = w // cell_size + 1

    grid_points = np.linspace( 0, w, cell_count)
    mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
    mapy = mapx.T

    mapx[1:-1,1:-1] = mapx[1:-1,1:-1] + randomex.random_normal( size=(cell_count-2, cell_count-2) )*(cell_size*0.24)
    mapy[1:-1,1:-1] = mapy[1:-1,1:-1] + randomex.random_normal( size=(cell_count-2, cell_count-2) )*(cell_size*0.24)

    half_cell_size = cell_size // 2

    mapx = cv2.resize(mapx, (w+cell_size,)*2 )[half_cell_size:-half_cell_size-1,half_cell_size:-half_cell_size-1].astype(np.float32)
    mapy = cv2.resize(mapy, (w+cell_size,)*2 )[half_cell_size:-half_cell_size-1,half_cell_size:-half_cell_size-1].astype(np.float32)

    #random transform
    random_transform_mat = cv2.getRotationMatrix2D((w // 2, w // 2), rotation, scale)
    random_transform_mat[:, 2] += (tx*w, ty*w)

    params = dict()
    params['mapx'] = mapx
    params['mapy'] = mapy
    params['rmat'] = random_transform_mat
    params['w'] = w
    params['flip'] = p_flip

    return params

def warp_by_params (params, img, can_warp, can_transform, can_flip, border_replicate, cv2_inter=cv2.INTER_CUBIC):
    if can_warp:
        img = cv2.remap(img, params['mapx'], params['mapy'], cv2_inter )
    if can_transform:
        img = cv2.warpAffine( img, params['rmat'], (params['w'], params['w']), borderMode=(cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT), flags=cv2_inter )
    if len(img.shape) == 2:
        img = img[...,None]
    if can_flip and params['flip']:
        img = img[:,::-1,...]
    return img