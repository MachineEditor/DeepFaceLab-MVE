from core.leras import nn
tf = nn.tf

class DeepFakeArchi(nn.ArchiBase):    
    """
    resolution
    
    mod     None - default
            'chervonij'
            'quick'
    """
    def __init__(self, resolution, mod=None):        
        super().__init__()
        
        if mod is None:
            class Downscale(nn.ModelBase):
                def __init__(self, in_ch, out_ch, kernel_size=5, dilations=1, subpixel=True, use_activator=True, *kwargs ):
                    self.in_ch = in_ch
                    self.out_ch = out_ch
                    self.kernel_size = kernel_size
                    self.dilations = dilations
                    self.subpixel = subpixel
                    self.use_activator = use_activator
                    super().__init__(*kwargs)

                def on_build(self, *args, **kwargs ):
                    self.conv1 = nn.Conv2D( self.in_ch,
                                            self.out_ch // (4 if self.subpixel else 1),
                                            kernel_size=self.kernel_size,
                                            strides=1 if self.subpixel else 2,
                                            padding='SAME', dilations=self.dilations)

                def forward(self, x):
                    x = self.conv1(x)
                    if self.subpixel:
                        x = nn.space_to_depth(x, 2)
                    if self.use_activator:
                        x = tf.nn.leaky_relu(x, 0.1)
                    return x

                def get_out_ch(self):
                    return (self.out_ch // 4) * 4

            class DownscaleBlock(nn.ModelBase):
                def on_build(self, in_ch, ch, n_downscales, kernel_size, dilations=1, subpixel=True):
                    self.downs = []

                    last_ch = in_ch
                    for i in range(n_downscales):
                        cur_ch = ch*( min(2**i, 8)  )
                        self.downs.append ( Downscale(last_ch, cur_ch, kernel_size=kernel_size, dilations=dilations, subpixel=subpixel) )
                        last_ch = self.downs[-1].get_out_ch()

                def forward(self, inp):
                    x = inp
                    for down in self.downs:
                        x = down(x)
                    return x

            class Upscale(nn.ModelBase):
                def on_build(self, in_ch, out_ch, kernel_size=3 ):
                    self.conv1 = nn.Conv2D( in_ch, out_ch*4, kernel_size=kernel_size, padding='SAME')

                def forward(self, x):
                    x = self.conv1(x)
                    x = tf.nn.leaky_relu(x, 0.1)
                    x = nn.depth_to_space(x, 2)
                    return x

            class ResidualBlock(nn.ModelBase):
                def on_build(self, ch, kernel_size=3 ):
                    self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')
                    self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')

                def forward(self, inp):
                    x = self.conv1(inp)
                    x = tf.nn.leaky_relu(x, 0.2)
                    x = self.conv2(x)
                    x = tf.nn.leaky_relu(inp + x, 0.2)
                    return x

            class UpdownResidualBlock(nn.ModelBase):
                def on_build(self, ch, inner_ch, kernel_size=3 ):
                    self.up   = Upscale (ch, inner_ch, kernel_size=kernel_size)
                    self.res  = ResidualBlock (inner_ch, kernel_size=kernel_size)
                    self.down = Downscale (inner_ch, ch, kernel_size=kernel_size, use_activator=False)

                def forward(self, inp):
                    x = self.up(inp)
                    x = upx = self.res(x)
                    x = self.down(x)
                    x = x + inp
                    x = tf.nn.leaky_relu(x, 0.2)
                    return x, upx

            class Encoder(nn.ModelBase):
                def on_build(self, in_ch, e_ch, is_hd):
                    self.is_hd=is_hd
                    if self.is_hd:
                        self.down1 = DownscaleBlock(in_ch, e_ch*2, n_downscales=4, kernel_size=3, dilations=1)
                        self.down2 = DownscaleBlock(in_ch, e_ch*2, n_downscales=4, kernel_size=5, dilations=1)
                        self.down3 = DownscaleBlock(in_ch, e_ch//2, n_downscales=4, kernel_size=5, dilations=2)
                        self.down4 = DownscaleBlock(in_ch, e_ch//2, n_downscales=4, kernel_size=7, dilations=2)
                    else:
                        self.down1 = DownscaleBlock(in_ch, e_ch, n_downscales=4, kernel_size=5, dilations=1, subpixel=False)

                def forward(self, inp):
                    if self.is_hd:
                        x = tf.concat([ nn.flatten(self.down1(inp)),
                                        nn.flatten(self.down2(inp)),
                                        nn.flatten(self.down3(inp)),
                                        nn.flatten(self.down4(inp)) ], -1 )
                    else:
                        x = nn.flatten(self.down1(inp))
                    return x
                    
            lowest_dense_res = resolution // 16

            class Inter(nn.ModelBase):
                def __init__(self, in_ch, ae_ch, ae_out_ch, is_hd=False, **kwargs):
                    self.in_ch, self.ae_ch, self.ae_out_ch = in_ch, ae_ch, ae_out_ch
                    super().__init__(**kwargs)

                def on_build(self):
                    in_ch, ae_ch, ae_out_ch = self.in_ch, self.ae_ch, self.ae_out_ch

                    self.dense1 = nn.Dense( in_ch, ae_ch )
                    self.dense2 = nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch )
                    self.upscale1 = Upscale(ae_out_ch, ae_out_ch)

                def forward(self, inp):
                    x = self.dense1(inp)
                    x = self.dense2(x)
                    x = nn.reshape_4D (x, lowest_dense_res, lowest_dense_res, self.ae_out_ch)
                    x = self.upscale1(x)
                    return x
                    
                @staticmethod
                def get_code_res():
                    return lowest_dense_res
                    
                def get_out_ch(self):
                    return self.ae_out_ch

            class Decoder(nn.ModelBase):
                def on_build(self, in_ch, d_ch, d_mask_ch, is_hd ):
                    self.is_hd = is_hd

                    self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
                    self.upscale1 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
                    self.upscale2 = Upscale(d_ch*4, d_ch*2, kernel_size=3)

                    if is_hd:
                        self.res0 = UpdownResidualBlock(in_ch, d_ch*8, kernel_size=3)
                        self.res1 = UpdownResidualBlock(d_ch*8, d_ch*4, kernel_size=3)
                        self.res2 = UpdownResidualBlock(d_ch*4, d_ch*2, kernel_size=3)
                        self.res3 = UpdownResidualBlock(d_ch*2, d_ch, kernel_size=3)
                    else:
                        self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
                        self.res1 = ResidualBlock(d_ch*4, kernel_size=3)
                        self.res2 = ResidualBlock(d_ch*2, kernel_size=3)

                    self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME')

                    self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
                    self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
                    self.upscalem2 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
                    self.out_convm = nn.Conv2D( d_mask_ch*2, 1, kernel_size=1, padding='SAME')

                def forward(self, inp):
                    z = inp

                    if self.is_hd:
                        x, upx = self.res0(z)
                        x = self.upscale0(x)
                        x = tf.nn.leaky_relu(x + upx, 0.2)
                        x, upx = self.res1(x)

                        x = self.upscale1(x)
                        x = tf.nn.leaky_relu(x + upx, 0.2)
                        x, upx = self.res2(x)

                        x = self.upscale2(x)
                        x = tf.nn.leaky_relu(x + upx, 0.2)
                        x, upx = self.res3(x)
                    else:
                        x = self.upscale0(z)
                        x = self.res0(x)
                        x = self.upscale1(x)
                        x = self.res1(x)
                        x = self.upscale2(x)
                        x = self.res2(x)

                    m = self.upscalem0(z)
                    m = self.upscalem1(m)
                    m = self.upscalem2(m)

                    return tf.nn.sigmoid(self.out_conv(x)), \
                           tf.nn.sigmoid(self.out_convm(m))
        
        elif mod == 'chervonij':
            class Downscale(nn.ModelBase):
                def __init__(self, in_ch, kernel_size=3, dilations=1, *kwargs ):
                    self.in_ch = in_ch
                    self.kernel_size = kernel_size
                    self.dilations = dilations
                    super().__init__(*kwargs)

                def on_build(self, *args, **kwargs ):
                    self.conv_base1 = nn.Conv2D( self.in_ch, self.in_ch//2, kernel_size=1, strides=1, padding='SAME', dilations=self.dilations)
                    self.conv_l1 = nn.Conv2D( self.in_ch//2, self.in_ch//2, kernel_size=self.kernel_size, strides=1, padding='SAME', dilations=self.dilations)
                    self.conv_l2 = nn.Conv2D( self.in_ch//2, self.in_ch//2, kernel_size=self.kernel_size, strides=2, padding='SAME', dilations=self.dilations)

                    self.conv_base2 = nn.Conv2D( self.in_ch, self.in_ch//2, kernel_size=1, strides=1, padding='SAME', dilations=self.dilations)
                    self.conv_r1 = nn.Conv2D( self.in_ch//2, self.in_ch//2, kernel_size=self.kernel_size, strides=2, padding='SAME', dilations=self.dilations)

                    self.pool_size = [1,1,2,2] if nn.data_format == 'NCHW' else [1,2,2,1]
                def forward(self, x):

                    x_l = self.conv_base1(x)
                    x_l = self.conv_l1(x_l)
                    x_l = self.conv_l2(x_l)

                    x_r = self.conv_base2(x)
                    x_r = self.conv_r1(x_r)

                    x_pool = tf.nn.max_pool(x, ksize=self.pool_size, strides=self.pool_size, padding='SAME', data_format=nn.data_format)

                    x = tf.concat([x_l, x_r, x_pool], axis=nn.conv2d_ch_axis)
                    x = tf.nn.leaky_relu(x, 0.1)
                    return x

            class Upscale(nn.ModelBase):
                def on_build(self, in_ch, out_ch, kernel_size=3 ):
                    self.conv1 = nn.Conv2D( in_ch, out_ch, kernel_size=kernel_size, padding='SAME')
                    self.conv2 = nn.Conv2D( out_ch, out_ch, kernel_size=kernel_size, padding='SAME')
                    self.conv3 = nn.Conv2D( out_ch, out_ch, kernel_size=kernel_size, padding='SAME')
                    self.conv4 = nn.Conv2D( out_ch, out_ch, kernel_size=kernel_size, padding='SAME')

                def forward(self, x):
                    x0 = self.conv1(x)
                    x1 = self.conv2(x0)
                    x2 = self.conv3(x1)
                    x3 = self.conv4(x2)
                    x = tf.concat([x0, x1, x2, x3], axis=nn.conv2d_ch_axis)
                    x = tf.nn.leaky_relu(x, 0.1)
                    x = nn.depth_to_space(x, 2)
                    return x

            class ResidualBlock(nn.ModelBase):
                def on_build(self, ch, kernel_size=3 ):
                    self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')
                    self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')
                    self.norm = nn.FRNorm2D(ch)

                def forward(self, inp):
                    x = self.conv1(inp)
                    x = tf.nn.leaky_relu(x, 0.2)
                    x = self.conv2(x)
                    x = self.norm(inp + x)
                    x = tf.nn.leaky_relu(x, 0.2)
                    return x

            class Encoder(nn.ModelBase):
                def on_build(self, in_ch, e_ch, **kwargs):
                    self.conv0 = nn.Conv2D(in_ch, e_ch, kernel_size=3, padding='SAME')

                    self.down0 = Downscale(e_ch)
                    self.down1 = Downscale(e_ch*2)
                    self.down2 = Downscale(e_ch*4)
                    self.down3 = Downscale(e_ch*8)
                    self.down4 = Downscale(e_ch*16)

                def forward(self, inp):
                    x = self.conv0(inp)
                    x = self.down0(x)
                    x = self.down1(x)
                    x = self.down2(x)
                    x = self.down3(x)
                    x = self.down4(x)
                    x = nn.flatten(x)
                    return x
            
            lowest_dense_res = resolution // 32
            
            class Inter(nn.ModelBase):
                def __init__(self, in_ch, ae_ch, ae_out_ch, **kwargs):
                    self.in_ch, self.ae_ch, self.ae_out_ch = in_ch, ae_ch, ae_out_ch
                    super().__init__(**kwargs)

                def on_build(self, **kwargs):
                    in_ch, ae_ch, ae_out_ch = self.in_ch, self.ae_ch, self.ae_out_ch

                    self.dense_l = nn.Dense( in_ch, ae_ch//2, kernel_initializer=tf.initializers.orthogonal)
                    self.dense_r = nn.Dense( in_ch, ae_ch//2, kernel_initializer=tf.initializers.orthogonal)#maxout_ch=4, 
                    self.dense = nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * (ae_out_ch//2), kernel_initializer=tf.initializers.orthogonal)
                    self.upscale1 = Upscale(ae_out_ch//2, ae_out_ch//2)

                def forward(self, inp):
                    x0 = self.dense_l(inp)
                    x1 = self.dense_r(inp)
                    x = tf.concat([x0, x1], axis=-1)
                    x = self.dense(x)
                    x = nn.reshape_4D (x, lowest_dense_res, lowest_dense_res, self.ae_out_ch//2)
                    x = self.upscale1(x)

                    return x

                def get_out_ch(self):
                    return self.ae_out_ch//2

            class Decoder(nn.ModelBase):
                def on_build(self, in_ch, d_ch, d_mask_ch, **kwargs):

                    self.upscale0 = Upscale(in_ch, d_ch*8)
                    self.upscale1 = Upscale(d_ch*8, d_ch*4)
                    self.upscale2 = Upscale(d_ch*4, d_ch*2)
                    self.upscale3 = Upscale(d_ch*2, d_ch)

                    self.res0 = ResidualBlock(d_ch*8)
                    self.res1 = ResidualBlock(d_ch*4)
                    self.res2 = ResidualBlock(d_ch*2)
                    self.res3 = ResidualBlock(d_ch)

                    self.out_conv  = nn.Conv2D( d_ch, 3, kernel_size=1, padding='SAME')

                    self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
                    self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
                    self.upscalem2 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
                    self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch, kernel_size=3)
                    self.out_convm = nn.Conv2D( d_mask_ch, 1, kernel_size=1, padding='SAME')

                def forward(self, inp):
                    z = inp

                    x = self.upscale0(z)
                    x = self.res0(x)
                    x = self.upscale1(x)
                    x = self.res1(x)
                    x = self.upscale2(x)
                    x = self.res2(x)
                    x = self.upscale3(x)
                    x = self.res3(x)

                    m = self.upscalem0(z)
                    m = self.upscalem1(m)
                    m = self.upscalem2(m)
                    m = self.upscalem3(m)

                    return tf.nn.sigmoid(self.out_conv(x)), \
                            tf.nn.sigmoid(self.out_convm(m))
        elif mod == 'quick':
            class Downscale(nn.ModelBase):
                def __init__(self, in_ch, out_ch, kernel_size=5, dilations=1, subpixel=True, use_activator=True, *kwargs ):
                    self.in_ch = in_ch
                    self.out_ch = out_ch
                    self.kernel_size = kernel_size
                    self.dilations = dilations
                    self.subpixel = subpixel
                    self.use_activator = use_activator
                    super().__init__(*kwargs)

                def on_build(self, *args, **kwargs ):
                    self.conv1 = nn.Conv2D( self.in_ch,
                                            self.out_ch // (4 if self.subpixel else 1),
                                            kernel_size=self.kernel_size,
                                            strides=1 if self.subpixel else 2,
                                            padding='SAME', dilations=self.dilations )

                def forward(self, x):
                    x = self.conv1(x)

                    if self.subpixel:
                        x = nn.space_to_depth(x, 2)

                    if self.use_activator:
                        x = nn.gelu(x)
                    return x

                def get_out_ch(self):
                    return (self.out_ch // 4) * 4

            class DownscaleBlock(nn.ModelBase):
                def on_build(self, in_ch, ch, n_downscales, kernel_size, dilations=1, subpixel=True):
                    self.downs = []

                    last_ch = in_ch
                    for i in range(n_downscales):
                        cur_ch = ch*( min(2**i, 8)  )
                        self.downs.append ( Downscale(last_ch, cur_ch, kernel_size=kernel_size, dilations=dilations, subpixel=subpixel) )
                        last_ch = self.downs[-1].get_out_ch()

                def forward(self, inp):
                    x = inp
                    for down in self.downs:
                        x = down(x)
                    return x

            class Upscale(nn.ModelBase):
                def on_build(self, in_ch, out_ch, kernel_size=3 ):
                    self.conv1 = nn.Conv2D( in_ch, out_ch*4, kernel_size=kernel_size, padding='SAME')

                def forward(self, x):
                    x = self.conv1(x)
                    x = nn.gelu(x)
                    x = nn.depth_to_space(x, 2)
                    return x

            class ResidualBlock(nn.ModelBase):
                def on_build(self, ch, kernel_size=3 ):
                    self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')
                    self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')

                def forward(self, inp):
                    x = self.conv1(inp)
                    x = nn.gelu(x)
                    x = self.conv2(x)
                    x = inp + x
                    x = nn.gelu(x)
                    return x

            class Encoder(nn.ModelBase):
                def on_build(self, in_ch, e_ch):
                    self.down1 = DownscaleBlock(in_ch, e_ch, n_downscales=4, kernel_size=5)
                def forward(self, inp):
                    return nn.flatten(self.down1(inp))

            lowest_dense_res = resolution // 16
            
            class Inter(nn.ModelBase):
                def __init__(self, in_ch,  ae_ch, ae_out_ch, d_ch, **kwargs):
                    self.in_ch, self.ae_ch, self.ae_out_ch, self.d_ch = in_ch, ae_ch, ae_out_ch, d_ch
                    super().__init__(**kwargs)

                def on_build(self):
                    in_ch, ae_ch, ae_out_ch, d_ch = self.in_ch, self.ae_ch, self.ae_out_ch, self.d_ch

                    self.dense1 = nn.Dense( in_ch, ae_ch, kernel_initializer=tf.initializers.orthogonal )
                    self.dense2 = nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch, kernel_initializer=tf.initializers.orthogonal )
                    self.upscale1 = Upscale(ae_out_ch, d_ch*8)
                    self.res1 = ResidualBlock(d_ch*8)

                def forward(self, inp):
                    x = self.dense1(inp)
                    x = self.dense2(x)
                    x = nn.reshape_4D (x, lowest_dense_res, lowest_dense_res, self.ae_out_ch)
                    x = self.upscale1(x)
                    x = self.res1(x)
                    return x

                def get_out_ch(self):
                    return self.ae_out_ch

            class Decoder(nn.ModelBase):
                def on_build(self, in_ch, d_ch):
                    self.upscale1 = Upscale(in_ch, d_ch*4)
                    self.res1     = ResidualBlock(d_ch*4)
                    self.upscale2 = Upscale(d_ch*4, d_ch*2)
                    self.res2     = ResidualBlock(d_ch*2)
                    self.upscale3 = Upscale(d_ch*2, d_ch*1)
                    self.res3     = ResidualBlock(d_ch*1)

                    self.upscalem1 = Upscale(in_ch, d_ch)
                    self.upscalem2 = Upscale(d_ch, d_ch//2)
                    self.upscalem3 = Upscale(d_ch//2, d_ch//2)

                    self.out_conv = nn.Conv2D( d_ch*1, 3, kernel_size=1, padding='SAME')
                    self.out_convm = nn.Conv2D( d_ch//2, 1, kernel_size=1, padding='SAME')

                def forward(self, inp):
                    z = inp
                    x = self.upscale1 (z)
                    x = self.res1     (x)
                    x = self.upscale2 (x)
                    x = self.res2     (x)
                    x = self.upscale3 (x)
                    x = self.res3     (x)

                    y = self.upscalem1 (z)
                    y = self.upscalem2 (y)
                    y = self.upscalem3 (y)

                    return tf.nn.sigmoid(self.out_conv(x)), \
                           tf.nn.sigmoid(self.out_convm(y))

        self.Encoder = Encoder
        self.Inter = Inter
        self.Decoder = Decoder

nn.DeepFakeArchi = DeepFakeArchi