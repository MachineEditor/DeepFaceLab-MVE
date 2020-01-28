def initialize_models(nn):
    tf = nn.tf
  

        
    class PatchDiscriminator(nn.ModelBase):
        def on_build(self, patch_size, in_ch, base_ch=256, kernel_initializer=None):
            prev_ch = in_ch
            self.convs = []            
            for i, (kernel_size, strides) in enumerate(patch_discriminator_kernels[patch_size]):
                cur_ch = base_ch * min( (2**i), 8 )
                self.convs.append ( nn.Conv2D( prev_ch, cur_ch, kernel_size=kernel_size, strides=strides, padding='SAME', kernel_initializer=kernel_initializer) )
                prev_ch = cur_ch

            self.out_conv =  nn.Conv2D( prev_ch, 1, kernel_size=1, padding='VALID', kernel_initializer=kernel_initializer)

        def forward(self, x):
            for conv in self.convs:
                x = tf.nn.leaky_relu( conv(x), 0.1 )
            return self.out_conv(x)
            
    nn.PatchDiscriminator = PatchDiscriminator
    
    
patch_discriminator_kernels = \
    { 1 : [ [1,1] ],             
      2 : [ [2,1] ],
      3 : [ [2,1], [2,1] ],        
      4 : [ [2,2], [2,2] ],        
      5 : [ [3,2], [2,2] ],        
      6 : [ [4,2], [2,2] ],
      7 : [ [3,2], [3,2] ],        
      8 : [ [4,2], [3,2] ],        
      9 : [ [3,2], [4,2] ], 
      10 : [ [4,2], [4,2] ], 
      11 : [ [3,2], [3,2], [2,1] ], 
      12 : [ [4,2], [3,2], [2,1] ], 
      13 : [ [3,2], [4,2], [2,1] ], 
      14 : [ [4,2], [4,2], [2,1] ], 
      15 : [ [3,2], [3,2], [3,1] ], 
      16 : [ [4,2], [3,2], [3,1] ] }