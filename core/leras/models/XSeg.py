from core.leras import nn
tf = nn.tf

class XSeg(nn.ModelBase):
    
    def on_build (self, in_ch, base_ch, out_ch):
        
        class ConvBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch):              
                self.conv = nn.Conv2D (in_ch, out_ch, kernel_size=3, padding='SAME')
                self.frn = nn.FRNorm2D(out_ch)
                self.tlu = nn.TLU(out_ch)

            def forward(self, x):                
                x = self.conv(x)
                x = self.frn(x)
                x = self.tlu(x)
                return x

        class UpConvBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch):
                self.conv = nn.Conv2DTranspose (in_ch, out_ch, kernel_size=3, padding='SAME')
                self.frn = nn.FRNorm2D(out_ch)
                self.tlu = nn.TLU(out_ch)

            def forward(self, x):
                x = self.conv(x)
                x = self.frn(x)
                x = self.tlu(x)
                return x

        self.conv01 = ConvBlock(in_ch, base_ch)
        self.conv02 = ConvBlock(base_ch, base_ch)
        self.bp0 = nn.BlurPool (filt_size=3)


        self.conv11 = ConvBlock(base_ch, base_ch*2)
        self.conv12 = ConvBlock(base_ch*2, base_ch*2)
        self.bp1 = nn.BlurPool (filt_size=3)

        self.conv21 = ConvBlock(base_ch*2, base_ch*4)
        self.conv22 = ConvBlock(base_ch*4, base_ch*4)
        self.conv23 = ConvBlock(base_ch*4, base_ch*4)
        self.bp2 = nn.BlurPool (filt_size=3)


        self.conv31 = ConvBlock(base_ch*4, base_ch*8)
        self.conv32 = ConvBlock(base_ch*8, base_ch*8)
        self.conv33 = ConvBlock(base_ch*8, base_ch*8)
        self.bp3 = nn.BlurPool (filt_size=3)

        self.conv41 = ConvBlock(base_ch*8, base_ch*8)
        self.conv42 = ConvBlock(base_ch*8, base_ch*8)
        self.conv43 = ConvBlock(base_ch*8, base_ch*8)
        self.bp4 = nn.BlurPool (filt_size=3)
        
        self.up4 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv43 = ConvBlock(base_ch*12, base_ch*8)
        self.uconv42 = ConvBlock(base_ch*8, base_ch*8)
        self.uconv41 = ConvBlock(base_ch*8, base_ch*8)

        self.up3 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv33 = ConvBlock(base_ch*12, base_ch*8)
        self.uconv32 = ConvBlock(base_ch*8, base_ch*8)
        self.uconv31 = ConvBlock(base_ch*8, base_ch*8)

        self.up2 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv23 = ConvBlock(base_ch*8, base_ch*4)
        self.uconv22 = ConvBlock(base_ch*4, base_ch*4)
        self.uconv21 = ConvBlock(base_ch*4, base_ch*4)

        self.up1 = UpConvBlock (base_ch*4, base_ch*2)
        self.uconv12 = ConvBlock(base_ch*4, base_ch*2)
        self.uconv11 = ConvBlock(base_ch*2, base_ch*2)

        self.up0 = UpConvBlock (base_ch*2, base_ch)
        self.uconv02 = ConvBlock(base_ch*2, base_ch)
        self.uconv01 = ConvBlock(base_ch, base_ch)
        self.out_conv = nn.Conv2D (base_ch, out_ch, kernel_size=3, padding='SAME')
        
        self.conv_center = ConvBlock(base_ch*8, base_ch*8)

    def forward(self, inp):
        x = inp

        x = self.conv01(x)
        x = x0 = self.conv02(x)
        x = self.bp0(x)

        x = self.conv11(x)
        x = x1 = self.conv12(x)
        x = self.bp1(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = x2 = self.conv23(x)
        x = self.bp2(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = x3 = self.conv33(x)
        x = self.bp3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = x4 = self.conv43(x)
        x = self.bp4(x)

        x = self.conv_center(x)

        x = self.up4(x)
        x = self.uconv43(tf.concat([x,x4],axis=nn.conv2d_ch_axis))
        x = self.uconv42(x)
        x = self.uconv41(x)

        x = self.up3(x)
        x = self.uconv33(tf.concat([x,x3],axis=nn.conv2d_ch_axis))
        x = self.uconv32(x)
        x = self.uconv31(x)

        x = self.up2(x)
        x = self.uconv23(tf.concat([x,x2],axis=nn.conv2d_ch_axis))
        x = self.uconv22(x)
        x = self.uconv21(x)

        x = self.up1(x)
        x = self.uconv12(tf.concat([x,x1],axis=nn.conv2d_ch_axis))
        x = self.uconv11(x)

        x = self.up0(x)
        x = self.uconv02(tf.concat([x,x0],axis=nn.conv2d_ch_axis))
        x = self.uconv01(x)

        logits = self.out_conv(x)
        return logits, tf.nn.sigmoid(logits)

nn.XSeg = XSeg