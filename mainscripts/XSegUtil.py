import traceback
import json
from pathlib import Path
import numpy as np

from core import pathex
from core.imagelib import IEPolys
from core.interact import interact as io
from DFLIMG import *


def merge(input_dir):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('input_dir not found. Please ensure it exists.')

    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)

    images_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        json_filepath = filepath.parent / (filepath.stem+'.json')
        if json_filepath.exists():
            dflimg = DFLIMG.load(filepath)
            if dflimg is not None and dflimg.has_data():
                try:
                    json_dict = json.loads(json_filepath.read_text())

                    seg_ie_polys = IEPolys()
                    total_points = 0
                    
                    #include polys first
                    for shape in json_dict['shapes']:
                        if shape['shape_type'] == 'polygon' and \
                           shape['label'] != '0':
                            seg_ie_poly = seg_ie_polys.add(1)

                            for x,y in shape['points']:
                                seg_ie_poly.add( int(x), int(y) )
                                total_points += 1
                                
                    #exclude polys
                    for shape in json_dict['shapes']:
                        if shape['shape_type'] == 'polygon' and \
                           shape['label'] == '0':
                            seg_ie_poly = seg_ie_polys.add(0)

                            for x,y in shape['points']:
                                seg_ie_poly.add( int(x), int(y) )
                                total_points += 1

                    if total_points == 0:
                        io.log_info(f"No points found in {json_filepath}, skipping.")
                        continue

                    dflimg.set_seg_ie_polys (seg_ie_polys)
                    dflimg.save()

                    json_filepath.unlink()

                    images_processed += 1
                except:
                    io.log_err(f"err {filepath}, {traceback.format_exc()}")
                    return

    io.log_info(f"Images processed: {images_processed}")

def split(input_dir ):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('input_dir not found. Please ensure it exists.')

    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)

    images_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        json_filepath = filepath.parent / (filepath.stem+'.json')
 
            
        dflimg = DFLIMG.load(filepath)
        if dflimg is not None and dflimg.has_data():
            try:
                seg_ie_polys = dflimg.get_seg_ie_polys()
                if seg_ie_polys is not None:                    
                    json_dict = {}
                    json_dict['version'] = "4.2.9"
                    json_dict['flags'] = {}
                    json_dict['shapes'] = []
                    json_dict['imagePath'] = filepath.name
                    json_dict['imageData'] = None
                    
                    for poly_type, points_list in seg_ie_polys:
                        shape_dict = {}
                        shape_dict['label'] = str(poly_type)
                        shape_dict['points'] = points_list
                        shape_dict['group_id'] = None
                        shape_dict['shape_type'] = 'polygon'
                        shape_dict['flags'] = {}
                        json_dict['shapes'].append( shape_dict )

                    json_filepath.write_text( json.dumps (json_dict,indent=4) )

                    dflimg.set_seg_ie_polys(None)
                    dflimg.save()
                    images_processed += 1
            except:
                io.log_err(f"err {filepath}, {traceback.format_exc()}")
                return

    io.log_info(f"Images processed: {images_processed}")