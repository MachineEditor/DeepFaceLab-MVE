from core.leras import nn
tf = nn.tf

class DeepFakeArchi(nn.ArchiBase):
    """
    resolution

    mod     None - default
            'quick'
    """
    def __init__(self, resolution, mod=None, opts=None):
        super().__init__()

        if opts is None:
            opts = ''

        if mod is None:
            class Downscale(nn.ModelBase):
                def __init__(self, in_ch, out_ch, kernel_size=5, *kwargs ):
                    self.in_ch = in_ch
                    self.out_ch = out_ch
                    self.kernel_size = kernel_size
                    super().__init__(*kwargs)

                def on_build(self, *args, **kwargs ):
                    self.conv1 = nn.Conv2D( self.in_ch, self.out_ch, kernel_size=self.kernel_size, strides=2, padding='SAME')

                def forward(self, x):
                    x = self.conv1(x)
                    x = tf.nn.leaky_relu(x, 0.1)
                    return x

                def get_out_ch(self):
                    return self.out_ch

            class DownscaleBlock(nn.ModelBase):
                def on_build(self, in_ch, ch, n_downscales, kernel_size):
                    self.downs = []

                    last_ch = in_ch
                    for i in range(n_downscales):
                        cur_ch = ch*( min(2**i, 8)  )
                        self.downs.append ( Downscale(last_ch, cur_ch, kernel_size=kernel_size) )
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

            class Encoder(nn.ModelBase):
                def on_build(self, in_ch, e_ch):
                    self.down1 = DownscaleBlock(in_ch, e_ch, n_downscales=4, kernel_size=5)

                def forward(self, inp):
                    return nn.flatten(self.down1(inp))

            lowest_dense_res = resolution // (32 if 'd' in opts else 16)

            class Inter(nn.ModelBase):
                def __init__(self, in_ch, ae_ch, ae_out_ch, **kwargs):
                    self.in_ch, self.ae_ch, self.ae_out_ch = in_ch, ae_ch, ae_out_ch
                    super().__init__(**kwargs)

                def on_build(self):
                    in_ch, ae_ch, ae_out_ch = self.in_ch, self.ae_ch, self.ae_out_ch
                    if 'u' in opts:
                        self.dense_norm = nn.DenseNorm()

                    self.dense1 = nn.Dense( in_ch, ae_ch )
                    self.dense2 = nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch )
                    self.upscale1 = Upscale(ae_out_ch, ae_out_ch)

                def forward(self, inp):
                    x = inp
                    if 'u' in opts:
                        x = self.dense_norm(x)
                    x = self.dense1(x)
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
                def on_build(self, in_ch, d_ch, d_mask_ch ):
                    self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
                    self.upscale1 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
                    self.upscale2 = Upscale(d_ch*4, d_ch*2, kernel_size=3)

                    self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
                    self.res1 = ResidualBlock(d_ch*4, kernel_size=3)
                    self.res2 = ResidualBlock(d_ch*2, kernel_size=3)

                    self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME')

                    self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
                    self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
                    self.upscalem2 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
                    self.out_convm = nn.Conv2D( d_mask_ch*2, 1, kernel_size=1, padding='SAME')

                    if 'd' in opts:
                        self.out_conv1  = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME')
                        self.out_conv2  = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME')
                        self.out_conv3  = nn.Conv2D( d_ch*2, 3, kernel_size=3, padding='SAME')
                        self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                        self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME')
                    else:
                        self.out_convm = nn.Conv2D( d_mask_ch*2, 1, kernel_size=1, padding='SAME')

                def forward(self, inp):
                    z = inp

                    x = self.upscale0(z)
                    x = self.res0(x)
                    x = self.upscale1(x)
                    x = self.res1(x)
                    x = self.upscale2(x)
                    x = self.res2(x)


                    if 'd' in opts:
                        x0 = tf.nn.sigmoid(self.out_conv(x))
                        x0 = nn.upsample2d(x0)
                        x1 = tf.nn.sigmoid(self.out_conv1(x))
                        x1 = nn.upsample2d(x1)
                        x2 = tf.nn.sigmoid(self.out_conv2(x))
                        x2 = nn.upsample2d(x2)
                        x3 = tf.nn.sigmoid(self.out_conv3(x))
                        x3 = nn.upsample2d(x3)

                        if nn.data_format == "NHWC":
                            tile_cfg = ( 1, resolution // 2, resolution //2, 1)
                        else:
                            tile_cfg = ( 1, 1, resolution // 2, resolution //2 )

                        z0 =  tf.concat ( ( tf.concat ( (  tf.ones ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ),
                                            tf.concat ( ( tf.zeros ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ) ), axis=nn.conv2d_spatial_axes[0] )

                        z0 = tf.tile ( z0, tile_cfg )

                        z1 =  tf.concat ( ( tf.concat ( ( tf.zeros ( (1,1,1,1) ), tf.ones ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ),
                                            tf.concat ( ( tf.zeros ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ) ), axis=nn.conv2d_spatial_axes[0] )
                        z1 = tf.tile ( z1, tile_cfg )

                        z2 =  tf.concat ( ( tf.concat ( (  tf.zeros ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ),
                                            tf.concat ( (  tf.ones ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ) ), axis=nn.conv2d_spatial_axes[0] )
                        z2 = tf.tile ( z2, tile_cfg )

                        z3 =  tf.concat ( ( tf.concat ( (  tf.zeros ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ),
                                            tf.concat ( (  tf.zeros ( (1,1,1,1) ), tf.ones ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ) ), axis=nn.conv2d_spatial_axes[0] )
                        z3 = tf.tile ( z3, tile_cfg )

                        x = x0*z0 + x1*z1 + x2*z2 + x3*z3
                    else:
                        x = tf.nn.sigmoid(self.out_conv(x))


                    m = self.upscalem0(z)
                    m = self.upscalem1(m)
                    m = self.upscalem2(m)
                    if 'd' in opts:
                        m = self.upscalem3(m)
                    m = tf.nn.sigmoid(self.out_convm(m))

                    return x, m
        
        self.Encoder = Encoder
        self.Inter = Inter
        self.Decoder = Decoder

nn.DeepFakeArchi = DeepFakeArchi