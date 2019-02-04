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
from facelib import LandmarksProcessor
from utils.SubprocessorBase import SubprocessorBase
import multiprocessing

def convert_png_to_jpg_file (filepath):
    filepath = Path(filepath)
    
    if filepath.suffix != '.png': 
        return  
    dflpng = DFLPNG.load (str(filepath), print_on_no_embedded_data=True)
    if dflpng is None: 
        return
    
    dfl_dict = dflpng.getDFLDictData()
    
    img = cv2.imread (str(filepath))
    new_filepath = str(filepath.parent / (filepath.stem + '.jpg'))
    cv2.imwrite ( new_filepath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    DFLJPG.embed_data( new_filepath, **dfl_dict )
    filepath.unlink()

        
def convert_png_to_jpg_folder (input_path):
    if not all(ord(c) < 128 for c in input_path):
        print ("Path to directory must contain only non unicode characters.")
        return
    input_path = Path(input_path)

    print ("Converting PNG to JPG...\r\n")

    for filepath in tqdm( Path_utils.get_image_paths(input_path), desc="Converting", ascii=True):
        filepath = Path(filepath)
        convert_png_to_jpg_file(filepath)
