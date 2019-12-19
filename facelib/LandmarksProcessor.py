import colorsys
import math
from enum import IntEnum

import cv2
import numpy as np
import numpy.linalg as npla

import imagelib
import mathlib
from facelib import FaceType
from imagelib import IEPolys
from mathlib.umeyama import umeyama

landmarks_2D = np.array([
[ 0.000213256,  0.106454  ], #17
[ 0.0752622,    0.038915  ], #18
[ 0.18113,      0.0187482 ], #19
[ 0.29077,      0.0344891 ], #20
[ 0.393397,     0.0773906 ], #21
[ 0.586856,     0.0773906 ], #22
[ 0.689483,     0.0344891 ], #23
[ 0.799124,     0.0187482 ], #24
[ 0.904991,     0.038915  ], #25
[ 0.98004,      0.106454  ], #26
[ 0.490127,     0.203352  ], #27
[ 0.490127,     0.307009  ], #28
[ 0.490127,     0.409805  ], #29
[ 0.490127,     0.515625  ], #30
[ 0.36688,      0.587326  ], #31
[ 0.426036,     0.609345  ], #32
[ 0.490127,     0.628106  ], #33
[ 0.554217,     0.609345  ], #34
[ 0.613373,     0.587326  ], #35
[ 0.121737,     0.216423  ], #36
[ 0.187122,     0.178758  ], #37
[ 0.265825,     0.179852  ], #38
[ 0.334606,     0.231733  ], #39
[ 0.260918,     0.245099  ], #40
[ 0.182743,     0.244077  ], #41
[ 0.645647,     0.231733  ], #42
[ 0.714428,     0.179852  ], #43
[ 0.793132,     0.178758  ], #44
[ 0.858516,     0.216423  ], #45
[ 0.79751,      0.244077  ], #46
[ 0.719335,     0.245099  ], #47
[ 0.254149,     0.780233  ], #48
[ 0.340985,     0.745405  ], #49
[ 0.428858,     0.727388  ], #50
[ 0.490127,     0.742578  ], #51
[ 0.551395,     0.727388  ], #52
[ 0.639268,     0.745405  ], #53
[ 0.726104,     0.780233  ], #54
[ 0.642159,     0.864805  ], #55
[ 0.556721,     0.902192  ], #56
[ 0.490127,     0.909281  ], #57
[ 0.423532,     0.902192  ], #58
[ 0.338094,     0.864805  ], #59
[ 0.290379,     0.784792  ], #60
[ 0.428096,     0.778746  ], #61
[ 0.490127,     0.785343  ], #62
[ 0.552157,     0.778746  ], #63
[ 0.689874,     0.784792  ], #64
[ 0.553364,     0.824182  ], #65
[ 0.490127,     0.831803  ], #66
[ 0.42689 ,     0.824182  ]  #67
], dtype=np.float32)


landmarks_2D_new = np.array([
[ 0.000213256,  0.106454  ], #17
[ 0.0752622,    0.038915  ], #18
[ 0.18113,      0.0187482 ], #19
[ 0.29077,      0.0344891 ], #20
[ 0.393397,     0.0773906 ], #21
[ 0.586856,     0.0773906 ], #22
[ 0.689483,     0.0344891 ], #23
[ 0.799124,     0.0187482 ], #24
[ 0.904991,     0.038915  ], #25
[ 0.98004,      0.106454  ], #26
[ 0.490127,     0.203352  ], #27
[ 0.490127,     0.307009  ], #28
[ 0.490127,     0.409805  ], #29
[ 0.490127,     0.515625  ], #30
[ 0.36688,      0.587326  ], #31
[ 0.426036,     0.609345  ], #32
[ 0.490127,     0.628106  ], #33
[ 0.554217,     0.609345  ], #34
[ 0.613373,     0.587326  ], #35
[ 0.121737,     0.216423  ], #36
[ 0.187122,     0.178758  ], #37
[ 0.265825,     0.179852  ], #38
[ 0.334606,     0.231733  ], #39
[ 0.260918,     0.245099  ], #40
[ 0.182743,     0.244077  ], #41
[ 0.645647,     0.231733  ], #42
[ 0.714428,     0.179852  ], #43
[ 0.793132,     0.178758  ], #44
[ 0.858516,     0.216423  ], #45
[ 0.79751,      0.244077  ], #46
[ 0.719335,     0.245099  ], #47
[ 0.254149,     0.780233  ], #48
[ 0.726104,     0.780233  ], #54
], dtype=np.float32)

# 68 point landmark definitions
landmarks_68_pt = { "mouth": (48,68),
                    "right_eyebrow": (17, 22),
                    "left_eyebrow": (22, 27),
                    "right_eye": (36, 42),
                    "left_eye": (42, 48),
                    "nose": (27, 36), # missed one point
                    "jaw": (0, 17) }


landmarks_68_3D = np.array( [
[-73.393523  , -29.801432   , 47.667532   ],
[-72.775014  , -10.949766   , 45.909403   ],
[-70.533638  , 7.929818     , 44.842580   ],
[-66.850058  , 26.074280    , 43.141114   ],
[-59.790187  , 42.564390    , 38.635298   ],
[-48.368973  , 56.481080    , 30.750622   ],
[-34.121101  , 67.246992    , 18.456453   ],
[-17.875411  , 75.056892    , 3.609035    ],
[0.098749    , 77.061286    , -0.881698   ],
[17.477031   , 74.758448    , 5.181201    ],
[32.648966   , 66.929021    , 19.176563   ],
[46.372358   , 56.311389    , 30.770570   ],
[57.343480   , 42.419126    , 37.628629   ],
[64.388482   , 25.455880    , 40.886309   ],
[68.212038   , 6.990805     , 42.281449   ],
[70.486405   , -11.666193   , 44.142567   ],
[71.375822   , -30.365191   , 47.140426   ],
[-61.119406  , -49.361602   , 14.254422   ],
[-51.287588  , -58.769795   , 7.268147    ],
[-37.804800  , -61.996155   , 0.442051    ],
[-24.022754  , -61.033399   , -6.606501   ],
[-11.635713  , -56.686759   , -11.967398  ],
[12.056636   , -57.391033   , -12.051204  ],
[25.106256   , -61.902186   , -7.315098   ],
[38.338588   , -62.777713   , -1.022953   ],
[51.191007   , -59.302347   , 5.349435    ],
[60.053851   , -50.190255   , 11.615746   ],
[0.653940    , -42.193790   , -13.380835  ],
[0.804809    , -30.993721   , -21.150853  ],
[0.992204    , -19.944596   , -29.284036  ],
[1.226783    , -8.414541    , -36.948060  ],
[-14.772472  , 2.598255     , -20.132003  ],
[-7.180239   , 4.751589     , -23.536684  ],
[0.555920    , 6.562900     , -25.944448  ],
[8.272499    , 4.661005     , -23.695741  ],
[15.214351   , 2.643046     , -20.858157  ],
[-46.047290  , -37.471411   , 7.037989    ],
[-37.674688  , -42.730510   , 3.021217    ],
[-27.883856  , -42.711517   , 1.353629    ],
[-19.648268  , -36.754742   , -0.111088   ],
[-28.272965  , -35.134493   , -0.147273   ],
[-38.082418  , -34.919043   , 1.476612    ],
[19.265868   , -37.032306   , -0.665746   ],
[27.894191   , -43.342445   , 0.247660    ],
[37.437529   , -43.110822   , 1.696435    ],
[45.170805   , -38.086515   , 4.894163    ],
[38.196454   , -35.532024   , 0.282961    ],
[28.764989   , -35.484289   , -1.172675   ],
[-28.916267  , 28.612716    , -2.240310   ],
[-17.533194  , 22.172187    , -15.934335  ],
[-6.684590   , 19.029051    , -22.611355  ],
[0.381001    , 20.721118    , -23.748437  ],
[8.375443    , 19.035460    , -22.721995  ],
[18.876618   , 22.394109    , -15.610679  ],
[28.794412   , 28.079924    , -3.217393   ],
[19.057574   , 36.298248    , -14.987997  ],
[8.956375    , 39.634575    , -22.554245  ],
[0.381549    , 40.395647    , -23.591626  ],
[-7.428895   , 39.836405    , -22.406106  ],
[-18.160634  , 36.677899    , -15.121907  ],
[-24.377490  , 28.677771    , -4.785684   ],
[-6.897633   , 25.475976    , -20.893742  ],
[0.340663    , 26.014269    , -22.220479  ],
[8.444722    , 25.326198    , -21.025520  ],
[24.474473   , 28.323008    , -5.712776   ],
[8.449166    , 30.596216    , -20.671489  ],
[0.205322    , 31.408738    , -21.903670  ],
[-7.198266   , 30.844876    , -20.328022  ] ], dtype=np.float32)

def convert_98_to_68(lmrks):
    #jaw
    result = [ lmrks[0] ]
    for i in range(2,16,2):
        result += [ ( lmrks[i] + (lmrks[i-1]+lmrks[i+1])/2 ) / 2  ]
    result += [ lmrks[16] ]
    for i in range(18,32,2):
        result += [ ( lmrks[i] + (lmrks[i-1]+lmrks[i+1])/2 ) / 2  ]
    result += [ lmrks[32] ]

    #eyebrows averaging
    result += [ lmrks[33],
                (lmrks[34]+lmrks[41])/2,
                (lmrks[35]+lmrks[40])/2,
                (lmrks[36]+lmrks[39])/2,
                (lmrks[37]+lmrks[38])/2,
              ]

    result += [ (lmrks[42]+lmrks[50])/2,
                (lmrks[43]+lmrks[49])/2,
                (lmrks[44]+lmrks[48])/2,
                (lmrks[45]+lmrks[47])/2,
                lmrks[46]
              ]

    #nose
    result += list ( lmrks[51:60] )

    #left eye (from our view)
    result += [ lmrks[60],
                lmrks[61],
                lmrks[63],
                lmrks[64],
                lmrks[65],
                lmrks[67] ]

    #right eye
    result += [ lmrks[68],
                lmrks[69],
                lmrks[71],
                lmrks[72],
                lmrks[73],
                lmrks[75] ]

    #mouth
    result += list ( lmrks[76:96] )

    return np.concatenate (result).reshape ( (68,2) )

def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform (mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points

def get_transform_mat (image_landmarks, output_size, face_type, scale=1.0):
    if not isinstance(image_landmarks, np.ndarray):
        image_landmarks = np.array (image_landmarks)

    """
    if face_type == FaceType.AVATAR:
        centroid = np.mean (image_landmarks, axis=0)

        mat = umeyama(image_landmarks[17:], landmarks_2D, True)[0:2]
        a, c = mat[0,0], mat[1,0]
        scale = math.sqrt((a * a) + (c * c))

        padding = (output_size / 64) * 32

        mat = np.eye ( 2,3 )
        mat[0,2] = -centroid[0]
        mat[1,2] = -centroid[1]
        mat = mat * scale * (output_size / 3)
        mat[:,2] += output_size / 2
    else:
    """
    remove_align = False
    if face_type == FaceType.FULL_NO_ALIGN:
        face_type = FaceType.FULL
        remove_align = True
    elif face_type == FaceType.HEAD_NO_ALIGN:
        face_type = FaceType.HEAD
        remove_align = True

    if face_type == FaceType.HALF:
        padding = 0
    elif face_type == FaceType.MID_FULL:
        padding = int(output_size * 0.06)
    elif face_type == FaceType.FULL:
        padding = (output_size / 64) * 12
    elif face_type == FaceType.HEAD:
        padding = (output_size / 64) * 21
    else:
        raise ValueError ('wrong face_type: ', face_type)

    #mat = umeyama(image_landmarks[17:], landmarks_2D, True)[0:2]
    mat = umeyama( np.concatenate ( [ image_landmarks[17:49] , image_landmarks[54:55] ] ) , landmarks_2D_new, True)[0:2]

    mat = mat * (output_size - 2 * padding)
    mat[:,2] += padding
    mat *= (1 / scale)
    mat[:,2] += -output_size*( ( (1 / scale) - 1.0 ) / 2 )

    if remove_align:
        bbox = transform_points ( [ (0,0), (0,output_size-1), (output_size-1, output_size-1), (output_size-1,0) ], mat, True)
        area = mathlib.polygon_area(bbox[:,0], bbox[:,1] )
        side = math.sqrt(area) / 2
        center = transform_points ( [(output_size/2,output_size/2)], mat, True)

        pts1 = np.float32([ center+[-side,-side], center+[side,-side], center+[-side,side] ])
        pts2 = np.float32([[0,0],[output_size-1,0],[0,output_size-1]])
        mat = cv2.getAffineTransform(pts1,pts2)

    return mat

def expand_eyebrows(lmrks, eyebrows_expand_mod=1.0):
    if len(lmrks) != 68:
        raise Exception('works only with 68 landmarks')
    lmrks = np.array( lmrks.copy(), dtype=np.int )

    # #nose
    ml_pnt = (lmrks[36] + lmrks[0]) // 2
    mr_pnt = (lmrks[16] + lmrks[45]) // 2

    # mid points between the mid points and eye
    ql_pnt = (lmrks[36] + ml_pnt) // 2
    qr_pnt = (lmrks[45] + mr_pnt) // 2

    # Top of the eye arrays
    bot_l = np.array((ql_pnt, lmrks[36], lmrks[37], lmrks[38], lmrks[39]))
    bot_r = np.array((lmrks[42], lmrks[43], lmrks[44], lmrks[45], qr_pnt))

    # Eyebrow arrays
    top_l = lmrks[17:22]
    top_r = lmrks[22:27]

    # Adjust eyebrow arrays
    lmrks[17:22] = top_l + eyebrows_expand_mod * 0.5 * (top_l - bot_l)
    lmrks[22:27] = top_r + eyebrows_expand_mod * 0.5 * (top_r - bot_r)
    return lmrks




def get_image_hull_mask (image_shape, image_landmarks, eyebrows_expand_mod=1.0, ie_polys=None, color=(1,) ):
    hull_mask = np.zeros(image_shape[0:2]+( len(color),),dtype=np.float32)

    lmrks = expand_eyebrows(image_landmarks, eyebrows_expand_mod)

    r_jaw = (lmrks[0:9], lmrks[17:18])
    l_jaw = (lmrks[8:17], lmrks[26:27])
    r_cheek = (lmrks[17:20], lmrks[8:9])
    l_cheek = (lmrks[24:27], lmrks[8:9])
    nose_ridge = (lmrks[19:25], lmrks[8:9],)
    r_eye = (lmrks[17:22], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    l_eye = (lmrks[22:27], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    nose = (lmrks[27:31], lmrks[31:36])
    parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

    for item in parts:
        merged = np.concatenate(item)
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(merged), color )

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)

    return hull_mask

def alpha_to_color (img_alpha, color):
    if len(img_alpha.shape) == 2:
        img_alpha = img_alpha[...,None]
    h,w,c = img_alpha.shape
    result = np.zeros( (h,w, len(color) ), dtype=np.float32 )
    result[:,:] = color

    return result * img_alpha



def get_cmask (image_shape, lmrks, eyebrows_expand_mod=1.0):
    h,w,c = image_shape

    hull = get_image_hull_mask (image_shape, lmrks, eyebrows_expand_mod, color=(1,) )

    result = np.zeros( (h,w,3), dtype=np.float32 )



    def process(w,h, data ):
        d = {}
        cur_lc = 0
        all_lines = []
        for s, pts_loop_ar in data:
            lines = []
            for pts, loop in pts_loop_ar:
                pts_len = len(pts)
                lines.append ( [ [ pts[i], pts[(i+1) % pts_len ] ]  for i in range(pts_len - (0 if loop else 1) ) ] )
            lines = np.concatenate (lines)

            lc = lines.shape[0]
            all_lines.append(lines)
            d[s] = cur_lc, cur_lc+lc
            cur_lc += lc
        all_lines = np.concatenate (all_lines, 0)

        #calculate signed distance for all points and lines
        line_count = all_lines.shape[0]
        pts_count = w*h

        all_lines = np.repeat ( all_lines[None,...], pts_count, axis=0 ).reshape ( (pts_count*line_count,2,2) )

        pts = np.empty( (h,w,line_count,2), dtype=np.float32 )
        pts[...,1] = np.arange(h)[:,None,None]
        pts[...,0] = np.arange(w)[:,None]
        pts = pts.reshape ( (h*w*line_count, -1) )

        a = all_lines[:,0,:]
        b = all_lines[:,1,:]
        pa = pts-a
        ba = b-a
        ph = np.clip ( np.einsum('ij,ij->i', pa, ba) / np.einsum('ij,ij->i', ba, ba), 0, 1 )
        dists = npla.norm ( pa - ba*ph[...,None], axis=1).reshape ( (h,w,line_count) )

        def get_dists(name, thickness=0):
            s,e = d[name]
            result = dists[...,s:e]
            if thickness != 0:
                result = np.abs(result)-thickness
            return np.min (result, axis=-1)

        return get_dists

    l_eye = lmrks[42:48]
    r_eye = lmrks[36:42]
    l_brow = lmrks[22:27]
    r_brow = lmrks[17:22]
    mouth = lmrks[48:60]

    up_nose = np.concatenate( (lmrks[27:31], lmrks[33:34]) )
    down_nose = lmrks[31:36]
    nose = np.concatenate ( (up_nose, down_nose) )

    gdf = process ( w,h,
                         (
                          ('eyes',  ((l_eye, True), (r_eye, True)) ),
                          ('brows', ((l_brow, False), (r_brow,False)) ),
                          ('up_nose', ((up_nose, False),) ),
                          ('down_nose', ((down_nose, False),) ),
                          ('mouth', ((mouth, True),) ),
                         )
                        )

    eyes_fall_dist = w // 32
    eyes_thickness = max( w // 64, 1 )

    brows_fall_dist = w // 32
    brows_thickness = max( w // 256, 1 )

    nose_fall_dist = w / 12
    nose_thickness = max( w // 96, 1 )

    mouth_fall_dist = w // 32
    mouth_thickness = max( w // 64, 1 )

    eyes_mask = gdf('eyes',eyes_thickness)
    eyes_mask = 1-np.clip( eyes_mask/ eyes_fall_dist, 0, 1)
    #eyes_mask = np.clip ( 1- ( np.sqrt( np.maximum(eyes_mask,0) ) / eyes_fall_dist ), 0, 1)
    #eyes_mask = np.clip ( 1- ( np.cbrt( np.maximum(eyes_mask,0) ) / eyes_fall_dist ), 0, 1)

    brows_mask = gdf('brows', brows_thickness)
    brows_mask = 1-np.clip( brows_mask / brows_fall_dist, 0, 1)
    #brows_mask = np.clip ( 1- ( np.sqrt( np.maximum(brows_mask,0) ) / brows_fall_dist ), 0, 1)

    mouth_mask = gdf('mouth', mouth_thickness)
    mouth_mask = 1-np.clip( mouth_mask / mouth_fall_dist, 0, 1)
    #mouth_mask = np.clip ( 1- ( np.sqrt( np.maximum(mouth_mask,0) ) / mouth_fall_dist ), 0, 1)

    def blend(a,b,k):
        x = np.clip ( 0.5+0.5*(b-a)/k, 0.0, 1.0 )
        return (a-b)*x+b - k*x*(1.0-x)


    #nose_mask = (a-b)*x+b - k*x*(1.0-x)

    #nose_mask = np.minimum (up_nose_mask , down_nose_mask )
    #nose_mask = 1-np.clip( nose_mask / nose_fall_dist, 0, 1)

    nose_mask = blend ( gdf('up_nose', nose_thickness), gdf('down_nose', nose_thickness), nose_thickness*3 )
    nose_mask = 1-np.clip( nose_mask / nose_fall_dist, 0, 1)

    up_nose_mask = gdf('up_nose', nose_thickness)
    up_nose_mask = 1-np.clip( up_nose_mask / nose_fall_dist, 0, 1)
    #up_nose_mask = np.clip ( 1- ( np.cbrt( np.maximum(up_nose_mask,0) ) / nose_fall_dist ), 0, 1)

    down_nose_mask = gdf('down_nose', nose_thickness)
    down_nose_mask = 1-np.clip( down_nose_mask / nose_fall_dist, 0, 1)
    #down_nose_mask = np.clip ( 1- ( np.cbrt( np.maximum(down_nose_mask,0) ) / nose_fall_dist ), 0, 1)

    #nose_mask = np.clip( up_nose_mask + down_nose_mask, 0, 1 )
    #nose_mask /= np.max(nose_mask)
    #nose_mask = np.maximum (up_nose_mask , down_nose_mask )
    #nose_mask = down_nose_mask

    #nose_mask = np.zeros_like(nose_mask)

    eyes_mask = eyes_mask * (1-mouth_mask)
    nose_mask = nose_mask * (1-eyes_mask)

    hull_mask = hull[...,0].copy()
    hull_mask = hull_mask * (1-eyes_mask) * (1-brows_mask) * (1-nose_mask) * (1-mouth_mask)

    #eyes_mask = eyes_mask * (1-nose_mask)

    mouth_mask= mouth_mask * (1-nose_mask)

    brows_mask = brows_mask * (1-nose_mask)* (1-eyes_mask )

    hull_mask = alpha_to_color(hull_mask, (0,1,0) )
    eyes_mask = alpha_to_color(eyes_mask, (1,0,0) )
    brows_mask = alpha_to_color(brows_mask, (0,0,1) )
    nose_mask = alpha_to_color(nose_mask, (0,1,1) )
    mouth_mask = alpha_to_color(mouth_mask, (0,0,1) )

    #nose_mask = np.maximum( up_nose_mask, down_nose_mask )

    result = hull_mask + mouth_mask+ nose_mask + brows_mask  + eyes_mask
    result *= hull
    #result = np.clip (result, 0, 1)
    return result

def get_image_eye_mask (image_shape, image_landmarks):
    if len(image_landmarks) != 68:
        raise Exception('get_image_eye_mask works only with 68 landmarks')

    hull_mask = np.zeros(image_shape[0:2]+(1,),dtype=np.float32)

    cv2.fillConvexPoly( hull_mask, cv2.convexHull( image_landmarks[36:42]), (1,) )
    cv2.fillConvexPoly( hull_mask, cv2.convexHull( image_landmarks[42:48]), (1,) )

    return hull_mask

def blur_image_hull_mask (hull_mask):

    maxregion = np.argwhere(hull_mask==1.0)
    miny,minx = maxregion.min(axis=0)[:2]
    maxy,maxx = maxregion.max(axis=0)[:2]
    lenx = maxx - minx;
    leny = maxy - miny;
    masky = int(minx+(lenx//2))
    maskx = int(miny+(leny//2))
    lowest_len = min (lenx, leny)
    ero = int( lowest_len * 0.085 )
    blur = int( lowest_len * 0.10 )

    hull_mask = cv2.erode(hull_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )
    hull_mask = cv2.blur(hull_mask, (blur, blur) )
    hull_mask = np.expand_dims (hull_mask,-1)

    return hull_mask

mirror_idxs = [
    [0,16],
    [1,15],
    [2,14],
    [3,13],
    [4,12],
    [5,11],
    [6,10],
    [7,9],

    [17,26],
    [18,25],
    [19,24],
    [20,23],
    [21,22],

    [36,45],
    [37,44],
    [38,43],
    [39,42],
    [40,47],
    [41,46],

    [31,35],
    [32,34],

    [50,52],
    [49,53],
    [48,54],
    [59,55],
    [58,56],
    [67,65],
    [60,64],
    [61,63] ]

def mirror_landmarks (landmarks, val):
    result = landmarks.copy()

    for idx in mirror_idxs:
        result [ idx ] = result [ idx[::-1] ]

    result[:,0] = val - result[:,0] - 1
    return result

def draw_landmarks (image, image_landmarks, color=(0,255,0), transparent_mask=False, ie_polys=None):
    if len(image_landmarks) != 68:
        raise Exception('get_image_eye_mask works only with 68 landmarks')

    int_lmrks = np.array(image_landmarks, dtype=np.int)

    jaw = int_lmrks[slice(*landmarks_68_pt["jaw"])]
    right_eyebrow = int_lmrks[slice(*landmarks_68_pt["right_eyebrow"])]
    left_eyebrow = int_lmrks[slice(*landmarks_68_pt["left_eyebrow"])]
    mouth = int_lmrks[slice(*landmarks_68_pt["mouth"])]
    right_eye = int_lmrks[slice(*landmarks_68_pt["right_eye"])]
    left_eye = int_lmrks[slice(*landmarks_68_pt["left_eye"])]
    nose = int_lmrks[slice(*landmarks_68_pt["nose"])]

    # open shapes
    cv2.polylines(image, tuple(np.array([v]) for v in ( right_eyebrow, jaw, left_eyebrow, np.concatenate((nose, [nose[-6]])) )),
                  False, color, lineType=cv2.LINE_AA)
    # closed shapes
    cv2.polylines(image, tuple(np.array([v]) for v in (right_eye, left_eye, mouth)),
                  True, color, lineType=cv2.LINE_AA)
    # the rest of the cicles
    for x, y in np.concatenate((right_eyebrow, left_eyebrow, mouth, right_eye, left_eye, nose), axis=0):
        cv2.circle(image, (x, y), 1, color, 1, lineType=cv2.LINE_AA)
    # jaw big circles
    for x, y in jaw:
        cv2.circle(image, (x, y), 2, color, lineType=cv2.LINE_AA)

    if transparent_mask:
        mask = get_image_hull_mask (image.shape, image_landmarks, ie_polys=ie_polys)
        image[...] = ( image * (1-mask) + image * mask / 2 )[...]

def draw_rect_landmarks (image, rect, image_landmarks, face_size, face_type, transparent_mask=False, ie_polys=None, landmarks_color=(0,255,0)):
    draw_landmarks(image, image_landmarks, color=landmarks_color, transparent_mask=transparent_mask, ie_polys=ie_polys)
    imagelib.draw_rect (image, rect, (255,0,0), 2 )

    image_to_face_mat = get_transform_mat (image_landmarks, face_size, face_type)
    points = transform_points ( [ (0,0), (0,face_size-1), (face_size-1, face_size-1), (face_size-1,0) ], image_to_face_mat, True)
    imagelib.draw_polygon (image, points, (0,0,255), 2)
    
    points = transform_points ( [ ( int(face_size*0.05), 0), ( int(face_size*0.1), int(face_size*0.1) ), ( 0, int(face_size*0.1) ) ], image_to_face_mat, True)
    imagelib.draw_polygon (image, points, (0,0,255), 2)

def calc_face_pitch(landmarks):
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array (landmarks)
    t = ( (landmarks[6][1]-landmarks[8][1]) + (landmarks[10][1]-landmarks[8][1]) ) / 2.0
    b = landmarks[8][1]
    return float(b-t)

def calc_face_yaw(landmarks):
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array (landmarks)
    l = ( (landmarks[27][0]-landmarks[0][0]) + (landmarks[28][0]-landmarks[1][0]) + (landmarks[29][0]-landmarks[2][0]) ) / 3.0
    r = ( (landmarks[16][0]-landmarks[27][0]) + (landmarks[15][0]-landmarks[28][0]) + (landmarks[14][0]-landmarks[29][0]) ) / 3.0
    return float(r-l)

#returns pitch,yaw,roll [-1...+1]
def estimate_pitch_yaw_roll(aligned_256px_landmarks):
    shape = (256,256)
    focal_length = shape[1]
    camera_center = (shape[1] / 2, shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, camera_center[0]],
         [0, focal_length, camera_center[1]],
         [0, 0, 1]], dtype=np.float32)

    (_, rotation_vector, translation_vector) = cv2.solvePnP(
        landmarks_68_3D,
        aligned_256px_landmarks.astype(np.float32),
        camera_matrix,
        np.zeros((4, 1)) )

    pitch, yaw, roll = mathlib.rotationMatrixToEulerAngles( cv2.Rodrigues(rotation_vector)[0] )
    pitch = np.clip ( pitch/1.30, -1.0, 1.0 )
    yaw = np.clip ( yaw / 1.11, -1.0, 1.0 )
    roll = np.clip ( roll/3.15, -1.0, 1.0 ) #todo radians
    return -pitch, yaw, roll
