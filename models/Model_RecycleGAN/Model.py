from models import ModelBase
import numpy as np
import cv2
from mathlib import get_power_of_two
from nnlib import nnlib


from facelib import FaceType
from samples import *

class Model(ModelBase):

    GAH5 = 'GA.h5'
    PAH5 = 'PA.h5'
    DAH5 = 'DA.h5'
    GBH5 = 'GB.h5'
    DBH5 = 'DB.h5'
    PBH5 = 'PB.h5'
    
    #override
    def onInitialize(self, batch_size=-1, **in_options):
        exec(nnlib.code_import_all, locals(), globals())

        created_batch_size = self.get_batch_size()
        if self.epoch == 0: 
            #first run
            
            print ("\nModel first run. Enter options.")
            
            try:
                created_resolution = int ( input ("Resolution (default:64, valid: 64,128,256) : ") )
            except:
                created_resolution = 64
                
            if created_resolution not in [64,128,256]:
                created_resolution = 64

            try:
                created_batch_size = int ( input ("Batch_size (minimum/default - 16) : ") )
            except:
                created_batch_size = 16
            created_batch_size = max(created_batch_size,1)
            
            print ("Done. If training won't start, decrease resolution")
     
            self.options['created_resolution'] = created_resolution
            self.options['created_batch_size'] = created_batch_size
            self.created_vram_gb = self.device_config.gpu_total_vram_gb
        else: 
            #not first run
            if 'created_batch_size' in self.options.keys():
                created_batch_size = self.options['created_batch_size']
            else:
                raise Exception("Continue training, but created_batch_size not found.")
                
            if 'created_resolution' in self.options.keys():
                created_resolution = self.options['created_resolution']
            else:
                raise Exception("Continue training, but created_resolution not found.")
        
        resolution = created_resolution
        bgr_shape = (resolution, resolution, 3)
        ngf = 64
        npf = 64
        ndf = 64
        lambda_A = 10
        lambda_B = 10
        
        self.set_batch_size(created_batch_size)
        
        use_batch_norm = created_batch_size > 1
        self.GA = modelify(ResNet (bgr_shape[2], use_batch_norm, n_blocks=6, ngf=ngf, use_dropout=False))(Input(bgr_shape))
        self.GB = modelify(ResNet (bgr_shape[2], use_batch_norm, n_blocks=6, ngf=ngf, use_dropout=False))(Input(bgr_shape))
        #self.GA = modelify(UNet (bgr_shape[2], use_batch_norm, num_downs=get_power_of_two(resolution)-1, ngf=ngf, use_dropout=True))(Input(bgr_shape))
        #self.GB = modelify(UNet (bgr_shape[2], use_batch_norm, num_downs=get_power_of_two(resolution)-1, ngf=ngf, use_dropout=True))(Input(bgr_shape))
        
        self.PA = modelify(UNetTemporalPredictor(bgr_shape[2], use_batch_norm, num_downs=get_power_of_two(resolution)-1, ngf=npf, use_dropout=True))([Input(bgr_shape), Input(bgr_shape)])
        self.PB = modelify(UNetTemporalPredictor(bgr_shape[2], use_batch_norm, num_downs=get_power_of_two(resolution)-1, ngf=npf, use_dropout=True))([Input(bgr_shape), Input(bgr_shape)])

        self.DA = modelify(NLayerDiscriminator(use_batch_norm, ndf=ndf, n_layers=3) ) (Input(bgr_shape))
        self.DB = modelify(NLayerDiscriminator(use_batch_norm, ndf=ndf, n_layers=3) ) (Input(bgr_shape))

        if not self.is_first_run():
            self.GA.load_weights (self.get_strpath_storage_for_file(self.GAH5))
            self.DA.load_weights (self.get_strpath_storage_for_file(self.DAH5))
            self.PA.load_weights (self.get_strpath_storage_for_file(self.PAH5))
            self.GB.load_weights (self.get_strpath_storage_for_file(self.GBH5))
            self.DB.load_weights (self.get_strpath_storage_for_file(self.DBH5))
            self.PB.load_weights (self.get_strpath_storage_for_file(self.PBH5))
            
        real_A0 = Input(bgr_shape, name="real_A0")
        real_A1 = Input(bgr_shape, name="real_A1")
        real_A2 = Input(bgr_shape, name="real_A2")
        
        real_B0 = Input(bgr_shape, name="real_B0")
        real_B1 = Input(bgr_shape, name="real_B1")
        real_B2 = Input(bgr_shape, name="real_B2")
        
        DA_ones =  K.ones ( K.int_shape(self.DA.outputs[0])[1:] )
        DA_zeros = K.zeros ( K.int_shape(self.DA.outputs[0])[1:] )
        DB_ones = K.ones ( K.int_shape(self.DB.outputs[0])[1:] )
        DB_zeros = K.zeros ( K.int_shape(self.DB.outputs[0])[1:] )

        def CycleLoss (t1,t2):
            return K.mean(K.square(t1 - t2))
        
        def RecurrentLOSS(t1,t2):
            return K.mean(K.square(t1 - t2))
            
        def RecycleLOSS(t1,t2):
            return K.mean(K.square(t1 - t2))
            
        fake_B0 = self.GA(real_A0)
        fake_B1 = self.GA(real_A1)
        
        fake_A0 = self.GB(real_B0)      
        fake_A1 = self.GB(real_B1)
        
        #rec_FB0 = self.GA(fake_A0)
        #rec_FB1 = self.GA(fake_A1)
        
        #rec_FA0 = self.GB(fake_B0)
        #rec_FA1 = self.GB(fake_B1)
 
        pred_A2 = self.PA ( [real_A0, real_A1])
        pred_B2 = self.PB ( [real_B0, real_B1])
        rec_A2 = self.GB ( self.PB ( [fake_B0, fake_B1]) )
        rec_B2 = self.GA ( self.PA ( [fake_A0, fake_A1]))
  
        loss_G = K.mean(K.square(self.DB(fake_B0) - DB_ones)) + \
                 K.mean(K.square(self.DB(fake_B1) - DB_ones)) + \
                 K.mean(K.square(self.DA(fake_A0) - DA_ones)) + \
                 K.mean(K.square(self.DA(fake_A1) - DA_ones)) + \
                 lambda_A * ( #CycleLoss(rec_FA0, real_A0) + \
                              #CycleLoss(rec_FA1, real_A1) + \
                              RecurrentLOSS(pred_A2, real_A2) + \
                              RecycleLOSS(rec_A2, real_A2) ) + \
                 lambda_B * ( #CycleLoss(rec_FB0, real_B0) + \
                              #CycleLoss(rec_FB1, real_B1) + \
                              RecurrentLOSS(pred_B2, real_B2) + \
                              RecycleLOSS(rec_B2, real_B2) )
        
        weights_G = self.GA.trainable_weights + self.GB.trainable_weights + self.PA.trainable_weights + self.PB.trainable_weights
        
        self.G_train = K.function ([real_A0, real_A1, real_A2, real_B0, real_B1, real_B2],[loss_G],
                                    Adam(lr=2e-4, beta_1=0.5, beta_2=0.999).get_updates(loss_G, weights_G) )
 
        ###########
        
        loss_D_A0 = ( K.mean(K.square( self.DA(real_A0) - DA_ones)) + \
                      K.mean(K.square( self.DA(fake_A0) - DA_zeros)) ) * 0.5
        
        loss_D_A1 = ( K.mean(K.square( self.DA(real_A1) - DA_ones)) + \
                      K.mean(K.square( self.DA(fake_A1) - DA_zeros)) ) * 0.5
                      
        loss_D_A = loss_D_A0 + loss_D_A1
        
        self.DA_train = K.function ([real_A0, real_A1, real_A2, real_B0, real_B1, real_B2],[loss_D_A],
                                    Adam(lr=2e-4, beta_1=0.5, beta_2=0.999).get_updates(loss_D_A, self.DA.trainable_weights) )
        
        ############
        
        loss_D_B0 = ( K.mean(K.square( self.DB(real_B0) - DB_ones)) + \
                      K.mean(K.square( self.DB(fake_B0) - DB_zeros)) ) * 0.5
        
        loss_D_B1 = ( K.mean(K.square( self.DB(real_B1) - DB_ones)) + \
                      K.mean(K.square( self.DB(fake_B1) - DB_zeros)) ) * 0.5
                      
        loss_D_B = loss_D_B0 + loss_D_B1
        
        self.DB_train = K.function ([real_A0, real_A1, real_A2, real_B0, real_B1, real_B2],[loss_D_B],
                                    Adam(lr=2e-4, beta_1=0.5, beta_2=0.999).get_updates(loss_D_B, self.DB.trainable_weights) )
        
        ############
        

        self.G_view = K.function([real_A0, real_A1, real_A2, real_B0, real_B1, real_B2],[fake_A0, fake_A1, pred_A2, rec_A2, fake_B0, fake_B1, pred_B2, rec_B2 ])
        self.G_convert = K.function([real_B0],[fake_A0])
        
        
        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            self.set_training_data_generators ([            
                    SampleGeneratorImageTemporal(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(random_flip = False, normalize_tanh = True), 
                        output_sample_types=[ [f.SOURCE | f.MODE_BGR, resolution] ] ),
                        
                    SampleGeneratorImageTemporal(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(random_flip = False, normalize_tanh = True), 
                        output_sample_types=[ [f.SOURCE | f.MODE_BGR, resolution] ] ),
                   ])
      
    #override
    def onSave(self):
        self.save_weights_safe( [[self.GA,    self.get_strpath_storage_for_file(self.GAH5)],
                                 [self.GB,    self.get_strpath_storage_for_file(self.GBH5)],
                                 [self.DA,    self.get_strpath_storage_for_file(self.DAH5)],
                                 [self.DB,    self.get_strpath_storage_for_file(self.DBH5)],
                                 [self.PA,    self.get_strpath_storage_for_file(self.PAH5)],
                                 [self.PB,    self.get_strpath_storage_for_file(self.PBH5)] ])
        
    #override
    def onTrainOneEpoch(self, sample):
        source_src_0, source_src_1, source_src_2, = sample[0]
        source_dst_0, source_dst_1, source_dst_2, = sample[1]        
     
        feed = [source_src_0, source_src_1, source_src_2, source_dst_0, source_dst_1, source_dst_2]

        loss_G,  = self.G_train ( feed )
        loss_DA, = self.DA_train( feed )
        loss_DB, = self.DB_train( feed )
        
        return ( ('G', loss_G), ('DA', loss_DA),  ('DB', loss_DB)  )

    #override
    def onGetPreview(self, sample):
        test_A0   = sample[0][0]
        test_A1   = sample[0][1]
        test_A2   = sample[0][2]
        
        test_B0   = sample[1][0]
        test_B1   = sample[1][1]
        test_B2   = sample[1][2]
        
        G_view_result = self.G_view([test_A0, test_A1, test_A2, test_B0, test_B1, test_B2])        

        fake_A0, fake_A1, pred_A2, rec_A2, fake_B0, fake_B1, pred_B2, rec_B2 = [ x[0] / 2 + 0.5 for x in G_view_result]        
        test_A0, test_A1, test_A2, test_B0, test_B1, test_B2 = [ x[0] / 2 + 0.5 for x in [test_A0, test_A1, test_A2, test_B0, test_B1, test_B2] ]
        
        
        r = np.concatenate ((np.concatenate ( (test_A0, test_A1, test_A2, pred_A2, fake_B0, fake_B1, rec_A2), axis=1),
                             np.concatenate ( (test_B0, test_B1, test_B2, pred_B2, fake_A0, fake_A1, rec_B2), axis=1)
                             ), axis=0)                            
                
        return [ ('RecycleGAN, A0-A1-A2-PA2-FB0-FB1-RA2, B0-B1-B2-PB2-FA0-FA1-RB2, ', r ) ]
    
    def predictor_func (self, face):
        x = self.G_convert ( [ np.expand_dims(face *2 - 1,0)]  )[0]
        return x[0] / 2 + 0.5
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterImage
                   
        return ConverterImage(self.predictor_func, predictor_input_size=self.options['created_resolution'], output_size=self.options['created_resolution'], **in_options)

