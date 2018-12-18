import os
import sys
import operator
import numpy as np
import cv2
from tqdm import tqdm
from shutil import copyfile

from pathlib import Path
from utils import Path_utils
from utils.AlignedPNG import AlignedPNG
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
    def onHostGetData(self):
        if len (self.input_data) > 0:
            return self.input_data.pop(0)    
        
        return None
    
    #override
    def onHostDataReturn (self, data):
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
        filename_path = Path( data[0] )
        image = cv2.imread( str(filename_path) )
        face_mask = None        
        
        a_png = AlignedPNG.load( str(filename_path) )        
        if a_png is not None:
            d = a_png.getFaceswapDictData()
            if (d is not None) and (d['landmarks'] is not None):            
                face_mask = LandmarksProcessor.get_image_hull_mask (image, np.array(d['landmarks']))
        
        if face_mask is not None:
            image = (image*face_mask).astype(np.uint8)
        else:
            print ( "%s - no embedded data found." % (str(filename_path)) ) 
            return [ str(filename_path), 0 ]
        
        return [ str(filename_path), estimate_sharpness( image ) ]

    #override
    def onClientGetDataName (self, data):
        #return string identificator of your data
        return data[0]
        
    #override
    def onHostResult (self, data, result):
        if result[1] == 0:
            filename_path = Path( data[0] )
            print ( "{0} - invalid image, renaming to {0}_invalid.".format(str(filename_path)) ) 
            filename_path.rename ( str(filename_path) + '_invalid' )
        else:
            self.result.append ( result )
        return 1
    
    #override    
    def onHostProcessEnd(self):
        pass
             
    #override
    def get_start_return(self):
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
    img_list = [ [x, np.mean ( cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2HSV)[...,2].flatten()  )] for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading") ]
    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)    
    return img_list
    
def sort_by_hue(input_path):
    print ("Sorting by hue...")
    img_list = [ [x, np.mean ( cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2HSV)[...,0].flatten()  )] for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading") ]
    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)    
    return img_list
    
def sort_by_face(input_path):

    print ("Sorting by face similarity...")

    img_list = []
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        filepath = Path(filepath)
        
        if filepath.suffix != '.png':
            print ("%s is not a png file required for sort_by_face" % (filepath.name) ) 
            continue
        
        a_png = AlignedPNG.load (str(filepath))
        if a_png is None:
            print ("%s failed to load" % (filepath.name) ) 
            continue
            
        d = a_png.getFaceswapDictData()
        
        if d is None or d['landmarks'] is None:          
            print ("%s - no embedded data found required for sort_by_face" % (filepath.name) )
            continue
        
        img_list.append( [str(filepath), np.array(d['landmarks']) ] )
        

    img_list_len = len(img_list)
    for i in tqdm ( range(0, img_list_len-1), desc="Sorting"):
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
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        filepath = Path(filepath)
        
        if filepath.suffix != '.png':
            print ("%s is not a png file required for sort_by_face_dissim" % (filepath.name) ) 
            continue
        
        a_png = AlignedPNG.load (str(filepath))
        if a_png is None:
            print ("%s failed to load" % (filepath.name) ) 
            continue
            
        d = a_png.getFaceswapDictData()
        
        if d is None or d['landmarks'] is None:          
            print ("%s - no embedded data found required for sort_by_face_dissim" % (filepath.name) )
            continue
        
        img_list.append( [str(filepath), np.array(d['landmarks']), 0 ] )
        
    img_list_len = len(img_list)
    for i in tqdm( range(0, img_list_len-1), desc="Sorting"):
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
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        filepath = Path(filepath)
        
        if filepath.suffix != '.png':
            print ("%s is not a png file required for sort_by_face_dissim" % (filepath.name) ) 
            continue
        
        a_png = AlignedPNG.load (str(filepath))
        if a_png is None:
            print ("%s failed to load" % (filepath.name) ) 
            continue
            
        d = a_png.getFaceswapDictData()
        
        if d is None or d['yaw_value'] is None:          
            print ("%s - no embedded data found required for sort_by_face_dissim" % (filepath.name) )
            continue
        
        img_list.append( [str(filepath), np.array(d['yaw_value']) ] )

    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
    
    return img_list
    
def sort_by_hist_blur(input_path):

    print ("Sorting by histogram similarity and blur...")

    img_list = []
    for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        img = cv2.imread(x)    
        img_list.append ([x, cv2.calcHist([img], [0], None, [256], [0, 256]),
                             cv2.calcHist([img], [1], None, [256], [0, 256]),
                             cv2.calcHist([img], [2], None, [256], [0, 256]),
                             estimate_sharpness(img)
                         ])

    img_list_len = len(img_list)
    for i in tqdm( range(0, img_list_len-1), desc="Sorting"):
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
     
    l = []
    for i in range(0, img_list_len-1):
        score = cv2.compareHist(img_list[i][1], img_list[i+1][1], cv2.HISTCMP_BHATTACHARYYA) + \
                cv2.compareHist(img_list[i][2], img_list[i+1][2], cv2.HISTCMP_BHATTACHARYYA) + \
                cv2.compareHist(img_list[i][3], img_list[i+1][3], cv2.HISTCMP_BHATTACHARYYA)
        l += [score]
    l = np.array(l)
    v = np.mean(l)
    if v*2 < np.max(l):
        v *= 2
    
    new_img_list = []
        
    start_group_i = 0
    odd_counter = 0
    for i in tqdm( range(0, img_list_len), desc="Sorting"):
        end_group_i = -1
        if i < img_list_len-1:
            score = cv2.compareHist(img_list[i][1], img_list[i+1][1], cv2.HISTCMP_BHATTACHARYYA) + \
                    cv2.compareHist(img_list[i][2], img_list[i+1][2], cv2.HISTCMP_BHATTACHARYYA) + \
                    cv2.compareHist(img_list[i][3], img_list[i+1][3], cv2.HISTCMP_BHATTACHARYYA)
                         
            if score >= v:
                end_group_i = i
                
        elif i == img_list_len-1:
            end_group_i = i
    
        if end_group_i >= start_group_i:
            odd_counter += 1
            
            s = sorted(img_list[start_group_i:end_group_i+1] , key=operator.itemgetter(4), reverse=True)         
            if odd_counter % 2 == 0:            
                new_img_list = new_img_list + s
            else:
                new_img_list = s + new_img_list
                
            start_group_i = i + 1

    return new_img_list
    
def sort_by_hist(input_path):

    print ("Sorting by histogram similarity...")

    img_list = []
    for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        img = cv2.imread(x)    
        img_list.append ([x, cv2.calcHist([img], [0], None, [256], [0, 256]),
                             cv2.calcHist([img], [1], None, [256], [0, 256]),
                             cv2.calcHist([img], [2], None, [256], [0, 256])
                         ])

    img_list_len = len(img_list)
    for i in tqdm( range(0, img_list_len-1), desc="Sorting"):
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
    def onHostGetData(self):
        if len (self.img_list_range) > 0:        
            return [self.img_list_range.pop(0)]
        
        return None
    
    #override
    def onHostDataReturn (self, data):
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
            score_total += cv2.compareHist(self.img_list[i][1], self.img_list[j][1], cv2.HISTCMP_BHATTACHARYYA) + \
                           cv2.compareHist(self.img_list[i][2], self.img_list[j][2], cv2.HISTCMP_BHATTACHARYYA) + \
                           cv2.compareHist(self.img_list[i][3], self.img_list[j][3], cv2.HISTCMP_BHATTACHARYYA)

        return score_total

    #override
    def onClientGetDataName (self, data):
        #return string identificator of your data
        return data[1]
        
    #override
    def onHostResult (self, data, result):
        self.img_list[data[0]][4] = result
        return 1
    
    #override    
    def onHostProcessEnd(self):
        pass
             
    #override
    def get_start_return(self):
        return self.img_list
        
def sort_by_hist_dissim(input_path):
    print ("Sorting by histogram dissimilarity...")

    img_list = []
    for x in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        img = cv2.imread(x)    
        img_list.append ([x, cv2.calcHist([img], [0], None, [256], [0, 256]),
                             cv2.calcHist([img], [1], None, [256], [0, 256]),
                             cv2.calcHist([img], [2], None, [256], [0, 256]), 0
                         ])

    img_list = HistDissimSubprocessor(img_list).process()
                         
    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(4), reverse=True)

    return img_list

def final_rename(input_path, img_list):
    for i in tqdm( range(0,len(img_list)), desc="Renaming" , leave=False):
        src = Path (img_list[i][0])        
        dst = input_path / ('%.5d_%s' % (i, src.name ))
        try:
            src.rename (dst)
        except:
            print ('fail to rename %s' % (src.name) )    
            
    for i in tqdm( range(0,len(img_list)) , desc="Renaming" ):
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
    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Loading"):
        filepath = Path(filepath)
        
        if filepath.suffix != '.png':
            print ("%s is not a png file required for sort_by_origname" % (filepath.name) ) 
            continue
        
        a_png = AlignedPNG.load (str(filepath))
        if a_png is None:
            print ("%s failed to load" % (filepath.name) ) 
            continue
            
        d = a_png.getFaceswapDictData()
        
        if d is None or d['source_filename'] is None:          
            print ("%s - no embedded data found required for sort_by_origname" % (filepath.name) )
            continue

        img_list.append( [str(filepath), d['source_filename']] )

    print ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1))
    return img_list
    
def main (input_path, sort_by_method):
    input_path = Path(input_path)
    sort_by_method = sort_by_method.lower()

    print ("Running sort tool.\r\n")
    
    img_list = []

    if sort_by_method == 'blur':            img_list = sort_by_blur (input_path)
    elif sort_by_method == 'face':          img_list = sort_by_face (input_path)
    elif sort_by_method == 'face-dissim':   img_list = sort_by_face_dissim (input_path)
    elif sort_by_method == 'face-yaw':      img_list = sort_by_face_yaw (input_path)
    elif sort_by_method == 'hist':          img_list = sort_by_hist (input_path)
    elif sort_by_method == 'hist-dissim':   img_list = sort_by_hist_dissim (input_path)
    elif sort_by_method == 'hist-blur':     img_list = sort_by_hist_blur (input_path)
    elif sort_by_method == 'brightness':    img_list = sort_by_brightness (input_path)
    elif sort_by_method == 'hue':           img_list = sort_by_hue (input_path)
    elif sort_by_method == 'origname':      img_list = sort_by_origname (input_path)       
    
    final_rename (input_path, img_list)