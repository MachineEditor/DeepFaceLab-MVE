import operator
from pathlib import Path

import cv2
import numpy as np



class FaceEnhancer(object):
    """
    x4 face enhancer
    """
    def __init__(self):
        from nnlib import nnlib
        exec( nnlib.import_all(), locals(), globals() )

        model_path = Path(__file__).parent / "FaceEnhancer.h5"
        if not model_path.exists():
            return
        
        bgr_inp = Input ( (192,192,3) )
        t_param_inp = Input ( (1,) )
        t_param1_inp = Input ( (1,) )
        x = Conv2D (64, 3, strides=1, padding='same' )(bgr_inp)
        
        a = Dense (64, use_bias=False) ( t_param_inp )
        a = Reshape( (1,1,64) )(a)
        b = Dense (64, use_bias=False ) ( t_param1_inp )
        b = Reshape( (1,1,64) )(b)    
        x = Add()([x,a,b])
        
        x = LeakyReLU(0.1)(x)

        x = LeakyReLU(0.1)(Conv2D (64, 3, strides=1, padding='same' )(x))
        x = e0 = LeakyReLU(0.1)(Conv2D (64, 3, strides=1, padding='same')(x))
        
        x = AveragePooling2D()(x)
        x = LeakyReLU(0.1)(Conv2D (112, 3, strides=1, padding='same')(x))
        x = e1 = LeakyReLU(0.1)(Conv2D (112, 3, strides=1, padding='same')(x))
        
        x = AveragePooling2D()(x)
        x = LeakyReLU(0.1)(Conv2D (192, 3, strides=1, padding='same')(x))
        x = e2 = LeakyReLU(0.1)(Conv2D (192, 3, strides=1, padding='same')(x))
        
        x = AveragePooling2D()(x)
        x = LeakyReLU(0.1)(Conv2D (336, 3, strides=1, padding='same')(x))
        x = e3 = LeakyReLU(0.1)(Conv2D (336, 3, strides=1, padding='same')(x))
        
        x = AveragePooling2D()(x)
        x = LeakyReLU(0.1)(Conv2D (512, 3, strides=1, padding='same')(x))
        x = e4 = LeakyReLU(0.1)(Conv2D (512, 3, strides=1, padding='same')(x))
        
        x = AveragePooling2D()(x)
        x = LeakyReLU(0.1)(Conv2D (512, 3, strides=1, padding='same')(x))
        x = LeakyReLU(0.1)(Conv2D (512, 3, strides=1, padding='same')(x))
        x = LeakyReLU(0.1)(Conv2D (512, 3, strides=1, padding='same')(x))
        x = LeakyReLU(0.1)(Conv2D (512, 3, strides=1, padding='same')(x))

        x = Concatenate()([ BilinearInterpolation()(x), e4 ])        

        x = LeakyReLU(0.1)(Conv2D (512, 3, strides=1, padding='same')(x))
        x = LeakyReLU(0.1)(Conv2D (512, 3, strides=1, padding='same')(x))
        
        x = Concatenate()([ BilinearInterpolation()(x), e3 ])
        x = LeakyReLU(0.1)(Conv2D (512, 3, strides=1, padding='same')(x))
        x = LeakyReLU(0.1)(Conv2D (512, 3, strides=1, padding='same')(x))
        
        x = Concatenate()([ BilinearInterpolation()(x), e2 ])
        x = LeakyReLU(0.1)(Conv2D (288, 3, strides=1, padding='same')(x))
        x = LeakyReLU(0.1)(Conv2D (288, 3, strides=1, padding='same')(x))
        
        x = Concatenate()([ BilinearInterpolation()(x), e1 ])
        x = LeakyReLU(0.1)(Conv2D (160, 3, strides=1, padding='same')(x))
        x = LeakyReLU(0.1)(Conv2D (160, 3, strides=1, padding='same')(x))
        
        x = Concatenate()([ BilinearInterpolation()(x), e0 ])
        x = LeakyReLU(0.1)(Conv2D (96, 3, strides=1, padding='same')(x))
        x = d0 = LeakyReLU(0.1)(Conv2D (96, 3, strides=1, padding='same')(x))

        x = LeakyReLU(0.1)(Conv2D (48, 3, strides=1, padding='same')(x))

        x = Conv2D (3, 3, strides=1, padding='same', activation='tanh')(x)
        out1x = Add()([bgr_inp, x])
        
        x = d0
        x = LeakyReLU(0.1)(Conv2D (96, 3, strides=1, padding='same')(x))
        x = LeakyReLU(0.1)(Conv2D (96, 3, strides=1, padding='same')(x))    
        x = d2x = BilinearInterpolation()(x)
        
        x = LeakyReLU(0.1)(Conv2D (48, 3, strides=1, padding='same')(x))
        x = Conv2D (3, 3, strides=1, padding='same', activation='tanh')(x)
        
        out2x = Add()([BilinearInterpolation()(out1x), x])
        
        x = d2x
        x = LeakyReLU(0.1)(Conv2D (72, 3, strides=1, padding='same')(x))
        x = LeakyReLU(0.1)(Conv2D (72, 3, strides=1, padding='same')(x))
        x = d4x = BilinearInterpolation()(x)
        
        x = LeakyReLU(0.1)(Conv2D (36, 3, strides=1, padding='same')(x))
        x = Conv2D (3, 3, strides=1, padding='same', activation='tanh')(x)
        out4x = Add()([BilinearInterpolation()(out2x), x ])

        self.model = keras.models.Model ( [bgr_inp,t_param_inp,t_param1_inp], [out4x] ) 
        self.model.load_weights (str(model_path))


    def enhance (self, inp_img, is_tanh=False, preserve_size=True):
        if not is_tanh:
            inp_img = np.clip( inp_img * 2 -1, -1, 1 )
            
        param = np.array([0.2])
        param1 = np.array([1.0])        
        up_res = 4
        patch_size = 192
        patch_size_half = patch_size // 2
    
        h,w,c = inp_img.shape
        
        i_max = w-patch_size+1
        j_max = h-patch_size+1     
        
        final_img = np.zeros ( (h*up_res,w*up_res,c), dtype=np.float32 )
        final_img_div = np.zeros ( (h*up_res,w*up_res,1), dtype=np.float32 )
 
        x = np.concatenate ( [ np.linspace (0,1,patch_size_half*up_res), np.linspace (1,0,patch_size_half*up_res) ] )
        x,y = np.meshgrid(x,x)
        patch_mask = (x*y)[...,None]
        
        j=0
        while j < j_max:
            i = 0
            while i < i_max:          
                patch_img = inp_img[j:j+patch_size, i:i+patch_size,:]             
                x = self.model.predict( [ patch_img[None,...], param, param1 ] )[0]
                final_img    [j*up_res:(j+patch_size)*up_res, i*up_res:(i+patch_size)*up_res,:] += x*patch_mask
                final_img_div[j*up_res:(j+patch_size)*up_res, i*up_res:(i+patch_size)*up_res,:] += patch_mask
                if i == i_max-1:
                    break
                i = min( i+patch_size_half, i_max-1)                
            if j == j_max-1:
                break
            j = min( j+patch_size_half, j_max-1)
            
        final_img_div[final_img_div==0] = 1.0
        final_img /= final_img_div
        
        if preserve_size:
            final_img = cv2.resize (final_img, (w,h), cv2.INTER_LANCZOS4)
        
        if not is_tanh:
            final_img = np.clip( final_img/2+0.5, 0, 1 )
            
        return final_img
