import cv2
from pathlib import Path
from utils import Path_utils
from utils.DFLPNG import DFLPNG
from utils.DFLJPG import DFLJPG
from utils.cv2_utils import *
from facelib import LandmarksProcessor
from interact import interact as io

def convert_png_to_jpg_file (filepath):
    filepath = Path(filepath)
    
    if filepath.suffix != '.png': 
        return
        
    dflpng = DFLPNG.load (str(filepath) )
    if dflpng is None:
        print ("%s is not a dfl image file" % (filepath.name) ) 
        return
    
    dfl_dict = dflpng.getDFLDictData()
    
    img = cv2_imread (str(filepath))
    new_filepath = str(filepath.parent / (filepath.stem + '.jpg'))
    cv2_imwrite ( new_filepath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    DFLJPG.embed_data( new_filepath, 
                       face_type=dfl_dict.get('face_type', None),
                       landmarks=dfl_dict.get('landmarks', None),
                       source_filename=dfl_dict.get('source_filename', None),
                       source_rect=dfl_dict.get('source_rect', None),
                       source_landmarks=dfl_dict.get('source_landmarks', None) )
                       
    filepath.unlink()
        
def convert_png_to_jpg_folder (input_path):
    input_path = Path(input_path)

    print ("Converting PNG to JPG...\r\n")

    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Converting"):
        filepath = Path(filepath)
        convert_png_to_jpg_file(filepath)
        
def add_landmarks_debug_images(input_path):
    print ("Adding landmarks debug images...")

    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)
 
        img = cv2_imread(str(filepath))
 
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None
            
        if dflimg is None:
            print ("%s is not a dfl image file" % (filepath.name) ) 
            continue

        if img is not None:
            face_landmarks = dflimg.get_landmarks()
            LandmarksProcessor.draw_landmarks(img, face_landmarks)
            
            output_file = '{}{}'.format( str(Path(str(input_path)) / filepath.stem),  '_debug.jpg')
            cv2_imwrite(output_file, img, [int(cv2.IMWRITE_JPEG_QUALITY), 50] )
            
