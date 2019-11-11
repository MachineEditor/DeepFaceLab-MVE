import traceback
from pathlib import Path

from interact import interact as io
from nnlib import DeepPortraitRelighting
from utils import Path_utils
from utils.cv2_utils import *
from utils.DFLJPG import DFLJPG
from utils.DFLPNG import DFLPNG


def relight(input_dir, lighten=None, random_one=None):
    if lighten is None:
        lighten = io.input_bool ("Lighten the faces? ( y/n default:n ) : ", False)

    if random_one is None:
        random_one = io.input_bool ("Relight the faces only with one random direction? ( y/n default:y ) : ", True)

    input_path = Path(input_dir)

    image_paths = [Path(x) for x in Path_utils.get_image_paths(input_path)]

    dpr = DeepPortraitRelighting()

    for filepath in io.progress_bar_generator(image_paths, "Relighting"):
        try:
            if filepath.suffix == '.png':
                dflimg = DFLPNG.load( str(filepath) )
            elif filepath.suffix == '.jpg':
                dflimg = DFLJPG.load ( str(filepath) )
            else:
                dflimg = None

            if dflimg is None:
                io.log_err ("%s is not a dfl image file" % (filepath.name) )
                continue
            else:
                if dflimg.get_relighted():
                    io.log_info (f"Skipping already relighted face [{filepath.name}]")
                    continue
                img = cv2_imread (str(filepath))

                if random_one:
                    relighted_imgs = dpr.relight_random(img,lighten=lighten)
                else:
                    relighted_imgs = dpr.relight_all(img,lighten=lighten)

                for i,relighted_img in enumerate(relighted_imgs):
                    im_flags = []
                    if filepath.suffix == '.jpg':
                        im_flags += [int(cv2.IMWRITE_JPEG_QUALITY), 100]

                    relighted_filename = filepath.parent / (filepath.stem+f'_relighted_{i}'+filepath.suffix)

                    cv2_imwrite (relighted_filename, relighted_img )
                    dflimg.embed_and_set (relighted_filename, source_filename="_", relighted=True )
        except:
            io.log_err (f"Exception occured while processing file {filepath.name}. Error: {traceback.format_exc()}")

def delete_relighted(input_dir):
    input_path = Path(input_dir)
    image_paths = [Path(x) for x in Path_utils.get_image_paths(input_path)]

    files_to_delete = []
    for filepath in io.progress_bar_generator(image_paths, "Loading"):
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is None:
            io.log_err ("%s is not a dfl image file" % (filepath.name) )
            continue
        else:
            if dflimg.get_relighted():
                files_to_delete += [filepath]

    for file in io.progress_bar_generator(files_to_delete, "Deleting"):
        file.unlink()
