import os
import sys
import operator
import numpy as np
import cv2
from tqdm import tqdm
from shutil import copyfile

from pathlib import Path
from utils import Path_utils
from utils import image_utils
from utils.DFLPNG import DFLPNG
from utils.DFLJPG import DFLJPG
from utils.cv2_utils import *
from facelib import LandmarksProcessor
from utils.SubprocessorBase import SubprocessorBase
import multiprocessing


def estimate_sharpness(image):       
    height, width = image.shape[:2]
    
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpness = 0
    for y in range(height):
        for x in range(width-1):        
            sharpness += abs( int(image[y, x]) - int(image[y, x+1]) )            
    
    for x in range(width):    
        for y in range(height-1):
            sharpness += abs( int(image[y, x]) - int(image[y+1, x]) )
    
    return sharpness
    

class BlurEstimatorSubprocessor(SubprocessorBase):
    #override
    def __init__(self, input_data ): 
        self.input_data = input_data
        self.result = []

        super().__init__('BlurEstimator', 60)           

    #override
    def onHostClientsInitialized(self):
        pass
        
    #override
    def process_info_generator(self):    
        for i in range(0, multiprocessing.cpu_count() ):
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i), 
                                      }

    #override
    def get_no_process_started_message(self):
        print ( 'Unable to start CPU processes.')
        
    #override
    def onHostGetProgressBarDesc(self):
        return None
        
    #override
    def onHostGetProgressBarLen(self):
        return len (self.input_data)
        
    #override
    def onHostGetData(self, host_dict):
        if len (self.input_data) > 0:
            return self.input_data.pop(0)    
        
        return None
    
    #override
    def onHostDataReturn (self, host_dict, data):
        self.input_data.insert(0, data)   
        
    #override
    def onClientInitialize(self, client_dict):
        self.safe_print ('Running on %s.' % (client_dict['device_name']) )
        return None

    #override
    def onClientFinalize(self):
        pass
        
    #override
    def onClientProcessData(self, data):
        filepath = Path( data[0] )
    
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath), print_on_no_embedded_data=True )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath), print_on_no_embedded_data=True )
        else:
            print ("%s is not a dfl image file" % (filepath.name) ) 
            dflimg = None   
            
        if dflimg is not None:
            image = cv2_imread( str(filepath) )
            image = ( image * \
                      LandmarksProcessor.get_image_hull_mask (image.shape, dflimg.get_landmarks()) \
                     ).astype(np.uint8)
            return [ str(filepath), estimate_sharpness( image ) ]
        else:
            return [ str(filepath), 0 ]

    #override
    def onClientGetDataName (self, data):
        #return string identificator of your data
        return data[0]
        
    #override
    def onHostResult (self, host_dict, data, result):
        if result[1] == 0:
            filename_path = Path( data[0] )
            print ( "{0} - invalid image, renaming to {0}_invalid.".format(str(filename_path)) ) 
            filename_path.rename ( str(filename_path) + '_invalid' )
        else:
            self.result.append ( result )
        return 1
    
    #override
    def onFinalizeAndGetResult(self):
        return self.result

    
def sort_by_blur(input_path):
    print ("Sorting by blur...")        
    
    img_list = [ (filename,[]) for filename in Path_utils.get_image_paths(input_path) ] 
    img_list = BlurEstimatorSubprocessor (img_list).process()
    
    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list
    
def sort_by_brightness(input_path):
    print ("Sorting by brightness...")
    img_list = [ [x, np.mean ( cv2.cvtColor(cv2_imread(x), cv2.COLOR_BGR2HSV)[...,2].flatten()  )] for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading", ascii=True) ]
    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)    
    return img_list
    
def sort_by_hue(input_path):
    print ("Sorting by hue...")
    img_list = [ [x, np.mean ( cv2.cvtColor(cv2_imread(x), cv2.COLOR_BGR2HSV)[...,0].flatten()  )] for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading", ascii=True) ]
    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)    
    return img_list
    
def sort_by_face(input_path):

    print ("Sorting by face similarity...")

    img_list = []
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading", ascii=True):
        filepath = Path(filepath)
        
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath), print_on_no_embedded_data=True )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath), print_on_no_embedded_data=True )
        else:
            print ("%s is not a dfl image file" % (filepath.name) ) 
            continue

        img_list.append( [str(filepath), dflimg.get_landmarks()] )
        

    img_list_len = len(img_list)
    for i in tqdm ( range(0, img_list_len-1), desc="Sorting", ascii=True):
        min_score = float("inf")
        j_min_score = i+1
        for j in range(i+1,len(img_list)):

            fl1 = img_list[i][1]
            fl2 = img_list[j][1]
            score = np.sum ( np.absolute ( (fl2 - fl1).flatten() ) )

            if score < min_score:
                min_score = score
                j_min_score = j
        img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]

    return img_list

def sort_by_face_dissim(input_path):

    print ("Sorting by face dissimilarity...")

    img_list = []
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading", ascii=True):
        filepath = Path(filepath)
        
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath), print_on_no_embedded_data=True )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath), print_on_no_embedded_data=True )
        else:
            print ("%s is not a dfl image file" % (filepath.name) ) 
            continue 
        
        img_list.append( [str(filepath), dflimg.get_landmarks(), 0 ] )
        
    img_list_len = len(img_list)
    for i in tqdm( range(0, img_list_len-1), desc="Sorting", ascii=True):
        score_total = 0
        for j in range(i+1,len(img_list)):
            if i == j:
                continue
            fl1 = img_list[i][1]
            fl2 = img_list[j][1]
            score_total += np.sum ( np.absolute ( (fl2 - fl1).flatten() ) )

        img_list[i][2] = score_total

    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)

    return img_list
    
def sort_by_face_yaw(input_path):
    print ("Sorting by face yaw...")
    img_list = []
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading", ascii=True):
        filepath = Path(filepath)
        
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath), print_on_no_embedded_data=True )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath), print_on_no_embedded_data=True )
        else:
            print ("%s is not a dfl image file" % (filepath.name) ) 
            continue
        
        pitch, yaw = LandmarksProcessor.estimate_pitch_yaw ( dflimg.get_landmarks() )
       
        img_list.append( [str(filepath), yaw ] )

    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
    
    return img_list
    
def sort_by_face_pitch(input_path):
    print ("Sorting by face pitch...")
    img_list = []
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading", ascii=True):
        filepath = Path(filepath)
        
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath), print_on_no_embedded_data=True )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath), print_on_no_embedded_data=True )
        else:
            print ("%s is not a dfl image file" % (filepath.name) ) 
            continue
        
        pitch, yaw = LandmarksProcessor.estimate_pitch_yaw ( dflimg.get_landmarks() )
       
        img_list.append( [str(filepath), pitch ] )

    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
    
    return img_list

class HistSsimSubprocessor(SubprocessorBase):
    #override
    def __init__(self, img_list ): 
        self.img_list = img_list
        self.img_list_len = len(img_list)
        
        slice_count = 20000
        sliced_count = self.img_list_len // slice_count
        
        if sliced_count > 12:
            sliced_count = 11.9
            slice_count = int(self.img_list_len / sliced_count)
            sliced_count = self.img_list_len // slice_count

        self.img_chunks_list = [ self.img_list[i*slice_count : (i+1)*slice_count] for i in range(sliced_count) ] + \
                               [ self.img_list[sliced_count*slice_count:] ]

        self.result = []

        super().__init__('HistSsim', 0)           

    #override
    def onHostClientsInitialized(self):
        pass
        
    #override
    def process_info_generator(self):    
        for i in range( len(self.img_chunks_list) ):
            yield 'CPU%d' % (i), {'i':i}, {'device_idx': i,
                                           'device_name': 'CPU%d' % (i)
                                          }

    #override
    def get_no_process_started_message(self):
        print ( 'Unable to start CPU processes.')
        
    #override
    def onHostGetProgressBarDesc(self):        
        return "Sorting"
        
    #override
    def onHostGetProgressBarLen(self):
        return len(self.img_list)
        
    #override
    def onHostClientsInitialized(self):
        self.inc_progress_bar(len(self.img_chunks_list))
        
    #override
    def onHostGetData(self, host_dict):     
        if len (self.img_chunks_list) > 0:        
            return self.img_chunks_list.pop(0)
        
        return None
    
    #override
    def onHostDataReturn (self, host_dict, data):
        raise Exception("Fail to process data. Decrease number of images and try again.")
        
    #override
    def onClientInitialize(self, client_dict):  
        self.safe_print ('Running on %s.' % (client_dict['device_name']) )
        return None

    #override
    def onClientFinalize(self):
        pass
        
    #override
    def onClientProcessData(self, data):            

        img_list = []
        for x in data:
            img = cv2_imread(x)    
            img_list.append ([x, cv2.calcHist([img], [0], None, [256], [0, 256]),
                                 cv2.calcHist([img], [1], None, [256], [0, 256]),
                                 cv2.calcHist([img], [2], None, [256], [0, 256])
                             ])
    
        img_list_len = len(img_list)
        for i in range(img_list_len-1):
            min_score = float("inf")
            j_min_score = i+1
            for j in range(i+1,len(img_list)):
                score = cv2.compareHist(img_list[i][1], img_list[j][1], cv2.HISTCMP_BHATTACHARYYA) + \
                        cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA) + \
                        cv2.compareHist(img_list[i][3], img_list[j][3], cv2.HISTCMP_BHATTACHARYYA)
                if score < min_score:
                    min_score = score
                    j_min_score = j
            img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]
            
            self.inc_progress_bar(1)
        
        return img_list     

    #override
    def onClientGetDataName (self, data):
        #return string identificator of your data
        return "Bunch of images"
        
    #override
    def onHostResult (self, host_dict, data, result):
        self.result += result
        return 0

    #override
    def onFinalizeAndGetResult(self):
        return self.result
    
def sort_by_hist(input_path):
    print ("Sorting by histogram similarity...")
    img_list = HistSsimSubprocessor(Path_utils.get_image_paths(input_path)).process()    
    return img_list

class HistDissimSubprocessor(SubprocessorBase):
    #override
    def __init__(self, img_list ): 
        self.img_list = img_list
        self.img_list_range = [i for i in range(0, len(img_list) )]

        self.result = []

        super().__init__('HistDissim', 60)           

    #override
    def onHostClientsInitialized(self):
        pass
        
    #override
    def process_info_generator(self):    
        for i in range(0, min(multiprocessing.cpu_count(), 8) ):
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i), 
                                      'img_list' : self.img_list
                                      }

    #override
    def get_no_process_started_message(self):
        print ( 'Unable to start CPU processes.')
        
    #override
    def onHostGetProgressBarDesc(self):
        return "Sorting"
        
    #override
    def onHostGetProgressBarLen(self):
        return len (self.img_list)
        
    #override
    def onHostGetData(self, host_dict):
        if len (self.img_list_range) > 0:        
            return [self.img_list_range.pop(0)]
        
        return None
    
    #override
    def onHostDataReturn (self, host_dict, data):
        self.img_list_range.insert(0, data[0])   
        
    #override
    def onClientInitialize(self, client_dict):
        self.img_list = client_dict['img_list']
        self.img_list_len = len(self.img_list)
        
        self.safe_print ('Running on %s.' % (client_dict['device_name']) )
        return None

    #override
    def onClientFinalize(self):
        pass
        
    #override
    def onClientProcessData(self, data):  
        i = data[0]
        score_total = 0
        for j in range( 0, self.img_list_len):
            if i == j:
                continue
            score_total += cv2.compareHist(self.img_list[i][1], self.img_list[j][1], cv2.HISTCMP_BHATTACHARYYA)

        return score_total

    #override
    def onClientGetDataName (self, data):
        #return string identificator of your data
        return self.img_list[data[0]][0]
        
    #override
    def onHostResult (self, host_dict, data, result):
        self.img_list[data[0]][2] = result
        return 1

    #override
    def onFinalizeAndGetResult(self):
        return self.img_list
        
def sort_by_hist_dissim(input_path):
    print ("Sorting by histogram dissimilarity...")

    img_list = []
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading", ascii=True):
        filepath = Path(filepath)
        
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath), print_on_no_embedded_data=True )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath), print_on_no_embedded_data=True )
        else:
            print ("%s is not a dfl image file" % (filepath.name) ) 
            continue
            
        image = cv2_imread(str(filepath))
        face_mask = LandmarksProcessor.get_image_hull_mask (image.shape, dflimg.get_landmarks())
        image = (image*face_mask).astype(np.uint8)

        img_list.append ([str(filepath), cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]), 0 ])

    img_list = HistDissimSubprocessor(img_list).process()
                         
    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)

    return img_list


class FinalLoaderSubprocessor(SubprocessorBase):
    #override
    def __init__(self, img_list ): 
        self.img_list = img_list

        self.result = []
        self.result_trash = []

        super().__init__('FinalLoader', 60)           

    #override
    def onHostClientsInitialized(self):
        pass
        
    #override
    def process_info_generator(self):    
        for i in range(0, min(multiprocessing.cpu_count(), 8) ):
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i)
                                      }

    #override
    def get_no_process_started_message(self):
        print ( 'Unable to start CPU processes.')
        
    #override
    def onHostGetProgressBarDesc(self):
        return "Loading"
        
    #override
    def onHostGetProgressBarLen(self):
        return len (self.img_list)
        
    #override
    def onHostGetData(self, host_dict):
        if len (self.img_list) > 0:        
            return [self.img_list.pop(0)]
        
        return None
    
    #override
    def onHostDataReturn (self, host_dict, data):
        self.img_list.insert(0, data[0])   
        
    #override
    def onClientInitialize(self, client_dict):        
        self.safe_print ('Running on %s.' % (client_dict['device_name']) )
        return None

    #override
    def onClientFinalize(self):
        pass
        
    #override
    def onClientProcessData(self, data):        
        filepath = Path(data[0])     

        try:
            if filepath.suffix == '.png':
                dflimg = DFLPNG.load( str(filepath), print_on_no_embedded_data=True )
            elif filepath.suffix == '.jpg':
                dflimg = DFLJPG.load( str(filepath), print_on_no_embedded_data=True )
            else:
                print ("%s is not a dfl image file" % (filepath.name) ) 
                raise Exception("")
            
            bgr = cv2_imread(str(filepath))
            if bgr is None:
                raise Exception ("Unable to load %s" % (filepath.name) ) 
                
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)        
            gray_masked = ( gray * LandmarksProcessor.get_image_hull_mask (bgr.shape, dflimg.get_landmarks() )[:,:,0] ).astype(np.uint8)
            sharpness = estimate_sharpness(gray_masked)
            pitch, yaw = LandmarksProcessor.estimate_pitch_yaw ( dflimg.get_landmarks() )
            
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        except Exception as e:
            print (e)
            return [ 1, [str(filepath)] ]
            
        return [ 0, [str(filepath), sharpness, hist, yaw ] ]
        

    #override
    def onClientGetDataName (self, data):
        #return string identificator of your data
        return data[0]
        
    #override
    def onHostResult (self, host_dict, data, result):
        if result[0] == 0:
            self.result.append (result[1])
        else:
            self.result_trash.append (result[1])
        return 1

    #override
    def onFinalizeAndGetResult(self):
        return self.result, self.result_trash
    
def sort_final(input_path):
    print ("Performing final sort.")
    
    img_list, trash_img_list = FinalLoaderSubprocessor( Path_utils.get_image_paths(input_path) ).process()
    final_img_list = []

    grads = 128
    imgs_per_grad = 15 

    grads_space = np.linspace (-1.0,1.0,grads)
    
    yaws_sample_list = [None]*grads
    for g in tqdm ( range(grads), desc="Sort by yaw", ascii=True ):    
        yaw = grads_space[g]
        next_yaw = grads_space[g+1] if g < grads-1 else yaw
        
        yaw_samples = []
        for img in img_list:
            s_yaw = -img[3]
            if (g == 0          and s_yaw < next_yaw) or \
               (g < grads-1     and s_yaw >= yaw and s_yaw < next_yaw) or \
               (g == grads-1    and s_yaw >= yaw):
                yaw_samples += [ img ]
        if len(yaw_samples) > 0:
            yaws_sample_list[g] = yaw_samples
    
    total_lack = 0
    for g in tqdm ( range (grads), desc="", ascii=True ):
        img_list = yaws_sample_list[g]
        img_list_len = len(img_list) if img_list is not None else 0
        
        lack = imgs_per_grad - img_list_len
        total_lack += max(lack, 0)        

    imgs_per_grad += total_lack // grads
    sharpned_imgs_per_grad = imgs_per_grad*10
    
    for g in tqdm ( range (grads), desc="Sort by blur", ascii=True ):
        img_list = yaws_sample_list[g]
        if img_list is None:
            continue

        img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)    
 
        if len(img_list) > imgs_per_grad*2:
            trash_img_list += img_list[len(img_list) // 2:]
            img_list = img_list[0: len(img_list) // 2]
        
        if len(img_list) > sharpned_imgs_per_grad:
            trash_img_list += img_list[sharpned_imgs_per_grad:]
            img_list = img_list[0:sharpned_imgs_per_grad]
            
        yaws_sample_list[g] = img_list
            
    for g in tqdm ( range (grads), desc="Sort by hist", ascii=True ):
        img_list = yaws_sample_list[g]
        if img_list is None:
            continue
            
        for i in range( len(img_list) ):
            score_total = 0
            for j in range( len(img_list) ):
                if i == j:
                    continue
                score_total += cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA)
            img_list[i][3] = score_total
            
        yaws_sample_list[g] = sorted(img_list, key=operator.itemgetter(3), reverse=True)    

    for g in tqdm ( range (grads), desc="Fetching best", ascii=True ):
        img_list = yaws_sample_list[g]
        if img_list is None:
            continue
            
        final_img_list += img_list[0:imgs_per_grad]
        trash_img_list += img_list[imgs_per_grad:]
    
    return final_img_list, trash_img_list
    
def sort_by_black(input_path):
    print ("Sorting by amount of black pixels...")

    img_list = []
    for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading", ascii=True):
        img = cv2_imread(x)
        img_list.append ([x, img[(img == 0)].size ])

    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=False)

    return img_list
    
def final_process(input_path, img_list, trash_img_list):
    if len(trash_img_list) != 0:
        parent_input_path = input_path.parent
        trash_path = parent_input_path / (input_path.stem + '_trash')
        trash_path.mkdir (exist_ok=True)
        
        print ("Trashing %d items to %s" % ( len(trash_img_list), str(trash_path) ) )        
        
        for filename in Path_utils.get_image_paths(trash_path):
            Path(filename).unlink()

        for i in tqdm( range(len(trash_img_list)), desc="Moving trash" , leave=False, ascii=True):
            src = Path (trash_img_list[i][0])        
            dst = trash_path / src.name
            try:
                src.rename (dst)
            except:
                print ('fail to trashing %s' % (src.name) )
                
        print ("")
        
    for i in tqdm( range(len(img_list)), desc="Renaming" , leave=False, ascii=True):
        src = Path (img_list[i][0])        
        dst = input_path / ('%.5d_%s' % (i, src.name ))
        try:
            src.rename (dst)
        except:
            print ('fail to rename %s' % (src.name) )
            
    for i in tqdm( range(len(img_list)) , desc="Renaming", ascii=True ):
        src = Path (img_list[i][0])
        
        src = input_path / ('%.5d_%s' % (i, src.name))
        dst = input_path / ('%.5d%s' % (i, src.suffix))
        try:
            src.rename (dst)
        except:
            print ('fail to rename %s' % (src.name) )   
        
def sort_by_origname(input_path):
    print ("Sort by original filename...")
    
    img_list = []
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading", ascii=True):
        filepath = Path(filepath)
        
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath), print_on_no_embedded_data=True )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load( str(filepath), print_on_no_embedded_data=True )
        else:
            print ("%s is not a dfl image file" % (filepath.name) ) 
            continue

        img_list.append( [str(filepath), dflimg.get_source_filename()] )

    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1))
    return img_list

def main (input_path, sort_by_method):
    input_path = Path(input_path)
    sort_by_method = sort_by_method.lower()

    print ("Running sort tool.\r\n")
    
    img_list = []
    trash_img_list = []
    if sort_by_method == 'blur':            img_list = sort_by_blur (input_path)
    elif sort_by_method == 'face':          img_list = sort_by_face (input_path)
    elif sort_by_method == 'face-dissim':   img_list = sort_by_face_dissim (input_path)
    elif sort_by_method == 'face-yaw':      img_list = sort_by_face_yaw (input_path)
    elif sort_by_method == 'face-pitch':    img_list = sort_by_face_pitch (input_path)
    elif sort_by_method == 'hist':          img_list = sort_by_hist (input_path)
    elif sort_by_method == 'hist-dissim':   img_list = sort_by_hist_dissim (input_path)
    elif sort_by_method == 'brightness':    img_list = sort_by_brightness (input_path)
    elif sort_by_method == 'hue':           img_list = sort_by_hue (input_path)
    elif sort_by_method == 'black':         img_list = sort_by_black (input_path)    
    elif sort_by_method == 'origname':      img_list = sort_by_origname (input_path)       
    elif sort_by_method == 'final':   img_list, trash_img_list = sort_final (input_path)  
    
    final_process (input_path, img_list, trash_img_list)
