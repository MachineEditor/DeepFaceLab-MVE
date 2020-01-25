import pickle
from pathlib import Path

import cv2

from DFLIMG import *
from facelib import LandmarksProcessor
from core.imagelib import IEPolys
from core.interact import interact as io
from core import pathex
from core.cv2ex import *


def save_faceset_metadata_folder(input_path):
    input_path = Path(input_path)

    metadata_filepath = input_path / 'meta.dat'

    io.log_info (f"Saving metadata to {str(metadata_filepath)}\r\n")

    d = {}
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)
        dflimg = DFLIMG.load (filepath)

        dfl_dict = dflimg.getDFLDictData()
        d[filepath.name] = ( dflimg.get_shape(), dfl_dict )

    try:
        with open(metadata_filepath, "wb") as f:
            f.write ( pickle.dumps(d) )
    except:
        raise Exception( 'cannot save %s' % (filename) )

    io.log_info("Now you can edit images.")
    io.log_info("!!! Keep same filenames in the folder.")
    io.log_info("You can change size of images, restoring process will downscale back to original size.")
    io.log_info("After that, use restore metadata.")

def restore_faceset_metadata_folder(input_path):
    input_path = Path(input_path)

    metadata_filepath = input_path / 'meta.dat'
    io.log_info (f"Restoring metadata from {str(metadata_filepath)}.\r\n")

    if not metadata_filepath.exists():
        io.log_err(f"Unable to find {str(metadata_filepath)}.")

    try:
        with open(metadata_filepath, "rb") as f:
            d = pickle.loads(f.read())
    except:
        raise FileNotFoundError(filename)

    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)

        shape, dfl_dict = d.get(filepath.name, None)

        img = cv2_imread (str(filepath))
        if img.shape != shape:
            img = cv2.resize (img, (shape[1], shape[0]), cv2.INTER_LANCZOS4 )

            if filepath.suffix == '.png':
                cv2_imwrite (str(filepath), img)
            elif filepath.suffix == '.jpg':
                cv2_imwrite (str(filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100] )

        if filepath.suffix == '.png':
            DFLPNG.embed_dfldict( str(filepath), dfl_dict )
        elif filepath.suffix == '.jpg':
            DFLJPG.embed_dfldict( str(filepath), dfl_dict )
        else:
            continue

    metadata_filepath.unlink()

def remove_ie_polys_file (filepath):
    filepath = Path(filepath)

    dflimg = DFLIMG.load (filepath)
    if dflimg is None:
        io.log_err ("%s is not a dfl image file" % (filepath.name) )
        return

    dflimg.remove_ie_polys()
    dflimg.embed_and_set( str(filepath) )


def remove_ie_polys_folder(input_path):
    input_path = Path(input_path)

    io.log_info ("Removing ie_polys...\r\n")

    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Removing"):
        filepath = Path(filepath)
        remove_ie_polys_file(filepath)

def remove_fanseg_file (filepath):
    filepath = Path(filepath)

    dflimg = DFLIMG.load (filepath)

    if dflimg is None:
        io.log_err ("%s is not a dfl image file" % (filepath.name) )
        return

    dflimg.remove_fanseg_mask()
    dflimg.embed_and_set( str(filepath) )


def remove_fanseg_folder(input_path):
    input_path = Path(input_path)

    io.log_info ("Removing fanseg mask...\r\n")

    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Removing"):
        filepath = Path(filepath)
        remove_fanseg_file(filepath)

def convert_png_to_jpg_file (filepath):
    filepath = Path(filepath)

    if filepath.suffix != '.png':
        return

    dflpng = DFLPNG.load (str(filepath) )
    if dflpng is None:
        io.log_err ("%s is not a dfl png image file" % (filepath.name) )
        return

    dfl_dict = dflpng.getDFLDictData()

    img = cv2_imread (str(filepath))
    new_filepath = str(filepath.parent / (filepath.stem + '.jpg'))
    cv2_imwrite ( new_filepath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    DFLJPG.embed_data( new_filepath,
                       face_type=dfl_dict.get('face_type', None),
                       landmarks=dfl_dict.get('landmarks', None),
                       ie_polys=dfl_dict.get('ie_polys', None),
                       source_filename=dfl_dict.get('source_filename', None),
                       source_rect=dfl_dict.get('source_rect', None),
                       source_landmarks=dfl_dict.get('source_landmarks', None) )

    filepath.unlink()

def convert_png_to_jpg_folder (input_path):
    input_path = Path(input_path)

    io.log_info ("Converting PNG to JPG...\r\n")

    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Converting"):
        filepath = Path(filepath)
        convert_png_to_jpg_file(filepath)

def add_landmarks_debug_images(input_path):
    io.log_info ("Adding landmarks debug images...")

    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)

        img = cv2_imread(str(filepath))

        dflimg = DFLIMG.load (filepath)

        if dflimg is None:
            io.log_err ("%s is not a dfl image file" % (filepath.name) )
            continue

        if img is not None:
            face_landmarks = dflimg.get_landmarks()
            LandmarksProcessor.draw_landmarks(img, face_landmarks, transparent_mask=True, ie_polys=IEPolys.load(dflimg.get_ie_polys()) )

            output_file = '{}{}'.format( str(Path(str(input_path)) / filepath.stem),  '_debug.jpg')
            cv2_imwrite(output_file, img, [int(cv2.IMWRITE_JPEG_QUALITY), 50] )

def recover_original_aligned_filename(input_path):
    io.log_info ("Recovering original aligned filename...")

    files = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        if dflimg is None:
            io.log_err ("%s is not a dfl image file" % (filepath.name) )
            continue

        files += [ [filepath, None, dflimg.get_source_filename(), False] ]

    files_len = len(files)
    for i in io.progress_bar_generator( range(files_len), "Sorting" ):
        fp, _, sf, converted = files[i]

        if converted:
            continue

        sf_stem = Path(sf).stem

        files[i][1] = fp.parent / ( sf_stem + '_0' + fp.suffix )
        files[i][3] = True
        c = 1

        for j in range(i+1, files_len):
            fp_j, _, sf_j, converted_j = files[j]
            if converted_j:
                continue

            if sf_j == sf:
                files[j][1] = fp_j.parent / ( sf_stem + ('_%d' % (c)) + fp_j.suffix )
                files[j][3] = True
                c += 1

    for file in io.progress_bar_generator( files, "Renaming", leave=False ):
        fs, _, _, _ = file
        dst = fs.parent / ( fs.stem + '_tmp' + fs.suffix )
        try:
            fs.rename (dst)
        except:
            io.log_err ('fail to rename %s' % (fs.name) )

    for file in io.progress_bar_generator( files, "Renaming" ):
        fs, fd, _, _ = file
        fs = fs.parent / ( fs.stem + '_tmp' + fs.suffix )
        try:
            fs.rename (fd)
        except:
            io.log_err ('fail to rename %s' % (fs.name) )
