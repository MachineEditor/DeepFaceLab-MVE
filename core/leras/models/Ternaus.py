"""
using https://github.com/ternaus/TernausNet
TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
"""

from core.leras import nn
tf = nn.tf

class Ternaus(nn.ModelBase):
    def on_build(self, in_ch, base_ch):

        self.features_0 = nn.Conv2D (in_ch, base_ch, kernel_size=3, padding='SAME')
        self.features_3 = nn.Conv2D (base_ch, base_ch*2, kernel_size=3, padding='SAME')                
        self.features_6 = nn.Conv2D (base_ch*2, base_ch*4, kernel_size=3, padding='SAME')
        self.features_8 = nn.Conv2D (base_ch*4, base_ch*4, kernel_size=3, padding='SAME')
        self.features_11 = nn.Conv2D (base_ch*4, base_ch*8, kernel_size=3, padding='SAME')
        self.features_13 = nn.Conv2D (base_ch*8, base_ch*8, kernel_size=3, padding='SAME')
        self.features_16 = nn.Conv2D (base_ch*8, base_ch*8, kernel_size=3, padding='SAME')
        self.features_18 = nn.Conv2D (base_ch*8, base_ch*8, kernel_size=3, padding='SAME')

        self.blurpool_0 = nn.BlurPool (filt_size=3) 
        self.blurpool_3 = nn.BlurPool (filt_size=3)
        self.blurpool_8 = nn.BlurPool (filt_size=3)                
        self.blurpool_13 = nn.BlurPool (filt_size=3)
        self.blurpool_18 = nn.BlurPool (filt_size=3)

        self.conv_center = nn.Conv2D (base_ch*8, base_ch*8, kernel_size=3, padding='SAME')

        self.conv1_up = nn.Conv2DTranspose (base_ch*8, base_ch*4, kernel_size=3, padding='SAME')
        self.conv1 = nn.Conv2D (base_ch*12, base_ch*8, kernel_size=3, padding='SAME')

        self.conv2_up = nn.Conv2DTranspose (base_ch*8, base_ch*4, kernel_size=3, padding='SAME')
        self.conv2 = nn.Conv2D (base_ch*12, base_ch*8, kernel_size=3, padding='SAME')

        self.conv3_up = nn.Conv2DTranspose (base_ch*8, base_ch*2, kernel_size=3, padding='SAME')
        self.conv3 = nn.Conv2D (base_ch*6, base_ch*4, kernel_size=3, padding='SAME')

        self.conv4_up = nn.Conv2DTranspose (base_ch*4, base_ch, kernel_size=3, padding='SAME')
        self.conv4 = nn.Conv2D (base_ch*3, base_ch*2, kernel_size=3, padding='SAME')

        self.conv5_up = nn.Conv2DTranspose (base_ch*2, base_ch//2, kernel_size=3, padding='SAME')
        self.conv5 = nn.Conv2D (base_ch//2+base_ch, base_ch, kernel_size=3, padding='SAME')

        self.out_conv = nn.Conv2D (base_ch, 1, kernel_size=3, padding='SAME')

    def forward(self, inp):
        x, = inp

        x = x0 = tf.nn.relu(self.features_0(x))
        x = self.blurpool_0(x)

        x = x1 = tf.nn.relu(self.features_3(x))
        x = self.blurpool_3(x)

        x = tf.nn.relu(self.features_6(x))
        x = x2 = tf.nn.relu(self.features_8(x))
        x = self.blurpool_8(x)

        x = tf.nn.relu(self.features_11(x))
        x = x3 = tf.nn.relu(self.features_13(x))
        x = self.blurpool_13(x)

        x = tf.nn.relu(self.features_16(x))
        x = x4 = tf.nn.relu(self.features_18(x))
        x = self.blurpool_18(x)

        x = self.conv_center(x)

        x = tf.nn.relu(self.conv1_up(x))
        x = tf.concat( [x,x4], nn.conv2d_ch_axis)
        x = tf.nn.relu(self.conv1(x))

        x = tf.nn.relu(self.conv2_up(x))
        x = tf.concat( [x,x3], nn.conv2d_ch_axis)
        x = tf.nn.relu(self.conv2(x))

        x = tf.nn.relu(self.conv3_up(x))
        x = tf.concat( [x,x2], nn.conv2d_ch_axis)
        x = tf.nn.relu(self.conv3(x))

        x = tf.nn.relu(self.conv4_up(x))
        x = tf.concat( [x,x1], nn.conv2d_ch_axis)
        x = tf.nn.relu(self.conv4(x))

        x = tf.nn.relu(self.conv5_up(x))
        x = tf.concat( [x,x0], nn.conv2d_ch_axis)
        x = tf.nn.relu(self.conv5(x))

        logits = self.out_conv(x)
        return logits, tf.nn.sigmoid(logits)
        
nn.Ternaus = Ternaus