import os
import traceback
from pathlib import Path

import cv2
import numpy as np
from numpy import linalg as npla

from facelib import FaceType, LandmarksProcessor
from nnlib import nnlib

"""
ported from https://github.com/1adrianb/face-alignment
"""
class FANExtractor(object):
    def __init__ (self):
        pass

    def __enter__(self):
        keras_model_path = Path(__file__).parent / "2DFAN-4.h5"
        if not keras_model_path.exists():
            return None

        exec( nnlib.import_all(), locals(), globals() )
        self.model = FANExtractor.BuildModel()
        self.model.load_weights(str(keras_model_path))

        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        del self.model
        return False #pass exception between __enter__ and __exit__ to outter level

    def extract (self, input_image, rects, second_pass_extractor=None, is_bgr=True, multi_sample=False):
        if len(rects) == 0:
            return []

        if is_bgr:
            input_image = input_image[:,:,::-1]
            is_bgr = False

        (h, w, ch) = input_image.shape

        landmarks = []
        for (left, top, right, bottom) in rects:
            scale = (right - left + bottom - top) / 195.0

            center = np.array( [ (left + right) / 2.0, (top + bottom) / 2.0] )
            centers = [ center ]

            if multi_sample:
                centers += [ center + [-1,-1],
                             center + [1,-1],
                             center + [1,1],
                             center + [-1,1],
                           ]

            images = []
            ptss = []

            try:
                for c in centers:
                    images += [ self.crop(input_image, c, scale)  ]

                images = np.stack (images)
                images = images.astype(np.float32) / 255.0                

                predicted = []
                for i in range( len(images) ):
                    predicted += [ self.model.predict ( images[i][None,...] ).transpose (0,3,1,2)[0] ]

                predicted = np.stack(predicted)                    

                for i, pred in enumerate(predicted):
                    ptss += [ self.get_pts_from_predict ( pred, centers[i], scale) ]
                pts_img = np.mean ( np.array(ptss), 0 )

                landmarks.append (pts_img)
            except:
                landmarks.append (None)

        if second_pass_extractor is not None:
            for i, lmrks in enumerate(landmarks):
                try:
                    if lmrks is not None:
                        image_to_face_mat = LandmarksProcessor.get_transform_mat (lmrks, 256, FaceType.FULL)
                        face_image = cv2.warpAffine(input_image, image_to_face_mat, (256, 256), cv2.INTER_CUBIC )

                        rects2 = second_pass_extractor.extract(face_image, is_bgr=is_bgr)
                        if len(rects2) == 1: #dont do second pass if faces != 1 detected in cropped image
                            lmrks2 = self.extract (face_image, [ rects2[0] ], is_bgr=is_bgr, multi_sample=True)[0]
                            landmarks[i] = LandmarksProcessor.transform_points (lmrks2, image_to_face_mat, True)
                except:
                    pass

        return landmarks

    def transform(self, point, center, scale, resolution):
        pt = np.array ( [point[0], point[1], 1.0] )
        h = 200.0 * scale
        m = np.eye(3)
        m[0,0] = resolution / h
        m[1,1] = resolution / h
        m[0,2] = resolution * ( -center[0] / h + 0.5 )
        m[1,2] = resolution * ( -center[1] / h + 0.5 )
        m = np.linalg.inv(m)
        return np.matmul (m, pt)[0:2]

    def crop(self, image, center, scale, resolution=256.0):
        ul = self.transform([1, 1], center, scale, resolution).astype( np.int )
        br = self.transform([resolution, resolution], center, scale, resolution).astype( np.int )

        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1] ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]

        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
        return newImg

    def get_pts_from_predict(self, a, center, scale):
        a_ch, a_h, a_w = a.shape

        b = a.reshape ( (a_ch, a_h*a_w) )
        c = b.argmax(1).reshape ( (a_ch, 1) ).repeat(2, axis=1).astype(np.float)
        c[:,0] %= a_w
        c[:,1] = np.apply_along_axis ( lambda x: np.floor(x / a_w), 0, c[:,1] )

        for i in range(a_ch):
            pX, pY = int(c[i,0]), int(c[i,1])
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array ( [a[i,pY,pX+1]-a[i,pY,pX-1], a[i,pY+1,pX]-a[i,pY-1,pX]] )
                c[i] += np.sign(diff)*0.25

        c += 0.5

        return np.array( [ self.transform (c[i], center, scale, a_w) for i in range(a_ch) ] )

    @staticmethod
    def BuildModel():
        def ConvBlock(out_planes, input):
            in_planes = K.int_shape(input)[-1]
            x = input
            x = BatchNormalization(momentum=0.1, epsilon=1e-05)(x)
            x = ReLU() (x)
            x = out1 = Conv2D( int(out_planes/2), kernel_size=3, strides=1, padding='valid', use_bias = False) (ZeroPadding2D(1)(x))

            x = BatchNormalization(momentum=0.1, epsilon=1e-05)(x)
            x = ReLU() (x)
            x = out2 = Conv2D( int(out_planes/4), kernel_size=3, strides=1, padding='valid', use_bias = False) (ZeroPadding2D(1)(x))

            x = BatchNormalization(momentum=0.1, epsilon=1e-05)(x)
            x = ReLU() (x)
            x = out3 = Conv2D( int(out_planes/4), kernel_size=3, strides=1, padding='valid', use_bias = False) (ZeroPadding2D(1)(x))

            x = Concatenate()([out1, out2, out3])

            if in_planes != out_planes:
                downsample = BatchNormalization(momentum=0.1, epsilon=1e-05)(input)
                downsample = ReLU() (downsample)
                downsample = Conv2D( out_planes, kernel_size=1, strides=1, padding='valid', use_bias = False) (downsample)
                x = Add ()([x, downsample])
            else:
                x = Add ()([x, input])


            return x

        def HourGlass (depth, input):
            up1 = ConvBlock(256, input)

            low1 = AveragePooling2D (pool_size=2, strides=2, padding='valid' )(input)
            low1 = ConvBlock (256, low1)

            if depth > 1:
                low2 = HourGlass (depth-1, low1)
            else:
                low2 = ConvBlock(256, low1)

            low3 = ConvBlock(256, low2)

            up2 = UpSampling2D(size=2) (low3)
            return Add() ( [up1, up2] )

        FAN_Input = Input ( (256, 256, 3) )

        x = FAN_Input

        x = Conv2D (64, kernel_size=7, strides=2, padding='valid')(ZeroPadding2D(3)(x))
        x = BatchNormalization(momentum=0.1, epsilon=1e-05)(x)
        x = ReLU()(x)

        x = ConvBlock (128, x)
        x = AveragePooling2D (pool_size=2, strides=2, padding='valid') (x)
        x = ConvBlock (128, x)
        x = ConvBlock (256, x)

        outputs = []
        previous = x
        for i in range(4):
            ll = HourGlass (4, previous)
            ll = ConvBlock (256, ll)

            ll = Conv2D(256, kernel_size=1, strides=1, padding='valid') (ll)
            ll = BatchNormalization(momentum=0.1, epsilon=1e-05)(ll)
            ll = ReLU() (ll)

            tmp_out = Conv2D(68, kernel_size=1, strides=1, padding='valid') (ll)
            outputs.append(tmp_out)

            if i < 4 - 1:
                ll = Conv2D(256, kernel_size=1, strides=1, padding='valid') (ll)
                previous = Add() ( [previous, ll, KL.Conv2D(256, kernel_size=1, strides=1, padding='valid') (tmp_out) ] )

        return Model(FAN_Input, outputs[-1] )
