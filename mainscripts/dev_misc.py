import traceback
import json
import multiprocessing
import shutil
from pathlib import Path
import cv2
import numpy as np

from core import imagelib, pathex
from core.cv2ex import *
from core.imagelib import IEPolys
from core.interact import interact as io
from core.joblib import Subprocessor
from core.leras import nn
from DFLIMG import *
from facelib import FaceType, LandmarksProcessor

from . import Extractor, Sorter
from .Extractor import ExtractSubprocessor


def extract_vggface2_dataset(input_dir, device_args={} ):
    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)

    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')

    bb_csv = input_path / 'loose_bb_train.csv'
    if not bb_csv.exists():
        raise ValueError('loose_bb_train.csv found. Please ensure it exists.')

    bb_lines = bb_csv.read_text().split('\n')
    bb_lines.pop(0)

    bb_dict = {}
    for line in bb_lines:
        name, l, t, w, h = line.split(',')
        name = name[1:-1]
        l, t, w, h = [ int(x) for x in (l, t, w, h) ]
        bb_dict[name] = (l,t,w, h)


    output_path = input_path.parent / (input_path.name + '_out')

    dir_names = pathex.get_all_dir_names(input_path)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    data = []
    for dir_name in io.progress_bar_generator(dir_names, "Collecting"):
        cur_input_path = input_path / dir_name
        cur_output_path = output_path / dir_name

        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)

        input_path_image_paths = pathex.get_image_paths(cur_input_path)

        for filename in input_path_image_paths:
            filename_path = Path(filename)

            name = filename_path.parent.name + '/' + filename_path.stem
            if name not in bb_dict:
                continue

            l,t,w,h = bb_dict[name]
            if min(w,h) < 128:
                continue

            data += [ ExtractSubprocessor.Data(filename=filename,rects=[ (l,t,l+w,t+h) ], landmarks_accurate=False, force_output_path=cur_output_path ) ]

    face_type = FaceType.fromString('full_face')

    io.log_info ('Performing 2nd pass...')
    data = ExtractSubprocessor (data, 'landmarks', 256, face_type, debug_dir=None, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False).run()

    io.log_info ('Performing 3rd pass...')
    ExtractSubprocessor (data, 'final', 256, face_type, debug_dir=None, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, final_output_path=None).run()


"""
    import code
    code.interact(local=dict(globals(), **locals()))

    data_len = len(data)
    i = 0
    while i < data_len-1:
        i_name = Path(data[i].filename).parent.name

        sub_data = []

        for j in range (i, data_len):
            j_name = Path(data[j].filename).parent.name
            if i_name == j_name:
                sub_data += [ data[j] ]
            else:
                break
        i = j

        cur_output_path = output_path / i_name

        io.log_info (f"Processing: {str(cur_output_path)}, {i}/{data_len} ")

        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)








    for dir_name in dir_names:

        cur_input_path = input_path / dir_name
        cur_output_path = output_path / dir_name

        input_path_image_paths = pathex.get_image_paths(cur_input_path)
        l = len(input_path_image_paths)
        #if l < 250 or l > 350:
        #    continue

        io.log_info (f"Processing: {str(cur_input_path)} ")

        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)


        data = []
        for filename in input_path_image_paths:
            filename_path = Path(filename)

            name = filename_path.parent.name + '/' + filename_path.stem
            if name not in bb_dict:
                continue

            bb = bb_dict[name]
            l,t,w,h = bb
            if min(w,h) < 128:
                continue

            data += [ ExtractSubprocessor.Data(filename=filename,rects=[ (l,t,l+w,t+h) ], landmarks_accurate=False ) ]



        io.log_info ('Performing 2nd pass...')
        data = ExtractSubprocessor (data, 'landmarks', 256, face_type, debug_dir=None, multi_gpu=False, cpu_only=False, manual=False).run()

        io.log_info ('Performing 3rd pass...')
        data = ExtractSubprocessor (data, 'final', 256, face_type, debug_dir=None, multi_gpu=False, cpu_only=False, manual=False, final_output_path=cur_output_path).run()


        io.log_info (f"Sorting: {str(cur_output_path)} ")
        Sorter.main (input_path=str(cur_output_path), sort_by_method='hist')

        import code
        code.interact(local=dict(globals(), **locals()))

        #try:
        #    io.log_info (f"Removing: {str(cur_input_path)} ")
        #    shutil.rmtree(cur_input_path)
        #except:
        #    io.log_info (f"unable to remove: {str(cur_input_path)} ")




def extract_vggface2_dataset(input_dir, device_args={} ):
    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)

    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')

    output_path = input_path.parent / (input_path.name + '_out')

    dir_names = pathex.get_all_dir_names(input_path)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)



    for dir_name in dir_names:

        cur_input_path = input_path / dir_name
        cur_output_path = output_path / dir_name

        l = len(pathex.get_image_paths(cur_input_path))
        if l < 250 or l > 350:
            continue

        io.log_info (f"Processing: {str(cur_input_path)} ")

        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)

        Extractor.main( str(cur_input_path),
              str(cur_output_path),
              detector='s3fd',
              image_size=256,
              face_type='full_face',
              max_faces_from_image=1,
              device_args=device_args )

        io.log_info (f"Sorting: {str(cur_input_path)} ")
        Sorter.main (input_path=str(cur_output_path), sort_by_method='hist')

        try:
            io.log_info (f"Removing: {str(cur_input_path)} ")
            shutil.rmtree(cur_input_path)
        except:
            io.log_info (f"unable to remove: {str(cur_input_path)} ")

"""

class CelebAMASKHQSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.masks_files_paths = client_dict['masks_files_paths']
            return None

        #override
        def process_data(self, data):
            filename = data[0]

            dflimg = DFLIMG.load(Path(filename))

            image_to_face_mat = dflimg.get_image_to_face_mat()
            src_filename = dflimg.get_source_filename()

            img = cv2_imread(filename)
            h,w,c = img.shape

            fanseg_mask = LandmarksProcessor.get_image_hull_mask(img.shape, dflimg.get_landmarks() )

            idx_name = '%.5d' % int(src_filename.split('.')[0])
            idx_files = [ x for x in self.masks_files_paths if idx_name in x ]

            skin_files = [ x for x in idx_files if 'skin' in x ]
            eye_glass_files = [ x for x in idx_files if 'eye_g' in x ]

            for files, is_invert in [ (skin_files,False),
                                      (eye_glass_files,True) ]:
                if len(files) > 0:
                    mask = cv2_imread(files[0])
                    mask = mask[...,0]
                    mask[mask == 255] = 1
                    mask = mask.astype(np.float32)
                    mask = cv2.resize(mask, (1024,1024) )
                    mask = cv2.warpAffine(mask, image_to_face_mat, (w, h), cv2.INTER_LANCZOS4)

                    if not is_invert:
                        fanseg_mask *= mask[...,None]
                    else:
                        fanseg_mask *= (1-mask[...,None])

            dflimg.embed_and_set (filename, fanseg_mask=fanseg_mask)
            return 1

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]

    #override
    def __init__(self, image_paths, masks_files_paths ):
        self.image_paths = image_paths
        self.masks_files_paths = masks_files_paths

        self.result = []
        super().__init__('CelebAMASKHQSubprocessor', CelebAMASKHQSubprocessor.Cli, 60)

    #override
    def process_info_generator(self):
        for i in range(min(multiprocessing.cpu_count(), 8)):
            yield 'CPU%d' % (i), {}, {'masks_files_paths' : self.masks_files_paths }

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Processing", len (self.image_paths))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.image_paths) > 0:
            return [self.image_paths.pop(0)]
        return None

    #override
    def on_data_return (self, host_dict, data):
        self.image_paths.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result

#unused in end user workflow
def apply_celebamaskhq(input_dir ):

    input_path = Path(input_dir)

    img_path = input_path / 'aligned'
    mask_path = input_path / 'mask'

    if not img_path.exists():
        raise ValueError(f'{str(img_path)} directory not found. Please ensure it exists.')

    CelebAMASKHQSubprocessor(pathex.get_image_paths(img_path),
                             pathex.get_image_paths(mask_path, subdirs=True) ).run()

    return

    paths_to_extract = []
    for filename in io.progress_bar_generator(pathex.get_image_paths(img_path), desc="Processing"):
        filepath = Path(filename)
        dflimg = DFLIMG.load(filepath)

        if dflimg is not None:
            paths_to_extract.append (filepath)

        image_to_face_mat = dflimg.get_image_to_face_mat()
        src_filename = dflimg.get_source_filename()

        #img = cv2_imread(filename)
        h,w,c = dflimg.get_shape()

        fanseg_mask = LandmarksProcessor.get_image_hull_mask( (h,w,c), dflimg.get_landmarks() )

        idx_name = '%.5d' % int(src_filename.split('.')[0])
        idx_files = [ x for x in masks_files if idx_name in x ]

        skin_files = [ x for x in idx_files if 'skin' in x ]
        eye_glass_files = [ x for x in idx_files if 'eye_g' in x ]

        for files, is_invert in [ (skin_files,False),
                                  (eye_glass_files,True) ]:

            if len(files) > 0:
                mask = cv2_imread(files[0])
                mask = mask[...,0]
                mask[mask == 255] = 1
                mask = mask.astype(np.float32)
                mask = cv2.resize(mask, (1024,1024) )
                mask = cv2.warpAffine(mask, image_to_face_mat, (w, h), cv2.INTER_LANCZOS4)

                if not is_invert:
                    fanseg_mask *= mask[...,None]
                else:
                    fanseg_mask *= (1-mask[...,None])

        #cv2.imshow("", (fanseg_mask*255).astype(np.uint8) )
        #cv2.waitKey(0)


        dflimg.embed_and_set (filename, fanseg_mask=fanseg_mask)


        #import code
        #code.interact(local=dict(globals(), **locals()))



#unused in end user workflow
def extract_fanseg(input_dir, device_args={} ):
    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)

    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')

    paths_to_extract = []
    for filename in pathex.get_image_paths(input_path) :
        filepath = Path(filename)
        dflimg = DFLIMG.load ( filepath )
        if dflimg is not None:
            paths_to_extract.append (filepath)

    paths_to_extract_len = len(paths_to_extract)
    if paths_to_extract_len > 0:
        io.log_info ("Performing extract fanseg for %d files..." % (paths_to_extract_len) )
        data = ExtractSubprocessor ([ ExtractSubprocessor.Data(filename) for filename in paths_to_extract ], 'fanseg', multi_gpu=multi_gpu, cpu_only=cpu_only).run()

#unused in end user workflow
def dev_test_68(input_dir ):
    # process 68 landmarks dataset with .pts files
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('input_dir not found. Please ensure it exists.')

    output_path = input_path.parent / (input_path.name+'_aligned')

    io.log_info(f'Output dir is % {output_path}')

    if output_path.exists():
        output_images_paths = pathex.get_image_paths(output_path)
        if len(output_images_paths) > 0:
            io.input_bool("WARNING !!! \n %s contains files! \n They will be deleted. \n Press enter to continue." % (str(output_path)), False )
            for filename in output_images_paths:
                Path(filename).unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    images_paths = pathex.get_image_paths(input_path)

    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        filepath = Path(filepath)


        pts_filepath = filepath.parent / (filepath.stem+'.pts')
        if pts_filepath.exists():
            pts = pts_filepath.read_text()
            pts_lines = pts.split('\n')

            lmrk_lines = None
            for pts_line in pts_lines:
                if pts_line == '{':
                    lmrk_lines = []
                elif pts_line == '}':
                    break
                else:
                    if lmrk_lines is not None:
                        lmrk_lines.append (pts_line)

            if lmrk_lines is not None and len(lmrk_lines) == 68:
                try:
                    lmrks = [ np.array ( lmrk_line.strip().split(' ') ).astype(np.float32).tolist() for lmrk_line in lmrk_lines]
                except Exception as e:
                    print(e)
                    print(filepath)
                    continue

                rect = LandmarksProcessor.get_rect_from_landmarks(lmrks)

                output_filepath = output_path / (filepath.stem+'.jpg')

                img = cv2_imread(filepath)
                img = imagelib.normalize_channels(img, 3)
                cv2_imwrite(output_filepath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95] )

                DFLJPG.embed_data(output_filepath, face_type=FaceType.toString(FaceType.MARK_ONLY),
                                                landmarks=lmrks,
                                                source_filename=filepath.name,
                                                source_rect=rect,
                                                source_landmarks=lmrks
                                    )

    io.log_info("Done.")

#unused in end user workflow
def extract_umd_csv(input_file_csv,
                    face_type='full_face',
                    device_args={} ):

    #extract faces from umdfaces.io dataset csv file with pitch,yaw,roll info.
    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)
    face_type = FaceType.fromString(face_type)

    input_file_csv_path = Path(input_file_csv)
    if not input_file_csv_path.exists():
        raise ValueError('input_file_csv not found. Please ensure it exists.')

    input_file_csv_root_path = input_file_csv_path.parent
    output_path = input_file_csv_path.parent / ('aligned_' + input_file_csv_path.name)

    io.log_info("Output dir is %s." % (str(output_path)) )

    if output_path.exists():
        output_images_paths = pathex.get_image_paths(output_path)
        if len(output_images_paths) > 0:
            io.input_bool("WARNING !!! \n %s contains files! \n They will be deleted. \n Press enter to continue." % (str(output_path)), False )
            for filename in output_images_paths:
                Path(filename).unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    try:
        with open( str(input_file_csv_path), 'r') as f:
            csv_file = f.read()
    except Exception as e:
        io.log_err("Unable to open or read file " + str(input_file_csv_path) + ": " + str(e) )
        return

    strings = csv_file.split('\n')
    keys = strings[0].split(',')
    keys_len = len(keys)
    csv_data = []
    for i in range(1, len(strings)):
        values = strings[i].split(',')
        if keys_len != len(values):
            io.log_err("Wrong string in csv file, skipping.")
            continue

        csv_data += [ { keys[n] : values[n] for n in range(keys_len) } ]

    data = []
    for d in csv_data:
        filename = input_file_csv_root_path / d['FILE']


        x,y,w,h = float(d['FACE_X']), float(d['FACE_Y']), float(d['FACE_WIDTH']), float(d['FACE_HEIGHT'])

        data += [ ExtractSubprocessor.Data(filename=filename, rects=[ [x,y,x+w,y+h] ]) ]

    images_found = len(data)
    faces_detected = 0
    if len(data) > 0:
        io.log_info ("Performing 2nd pass from csv file...")
        data = ExtractSubprocessor (data, 'landmarks', multi_gpu=multi_gpu, cpu_only=cpu_only).run()

        io.log_info ('Performing 3rd pass...')
        data = ExtractSubprocessor (data, 'final', face_type, None, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, final_output_path=output_path).run()
        faces_detected += sum([d.faces_detected for d in data])


    io.log_info ('-------------------------')
    io.log_info ('Images found:        %d' % (images_found) )
    io.log_info ('Faces detected:      %d' % (faces_detected) )
    io.log_info ('-------------------------')

def dev_test1(input_dir):
    input_path = Path(input_dir)

    dir_names = pathex.get_all_dir_names(input_path)

    for dir_name in io.progress_bar_generator(dir_names, desc="Processing"):

        img_paths = pathex.get_image_paths (input_path / dir_name)
        for filename in img_paths:
            filepath = Path(filename)

            dflimg = DFLIMG.load (filepath)
            if dflimg is None:
                raise ValueError

            dflimg.embed_and_set(filename, person_name=dir_name)

            #import code
            #code.interact(local=dict(globals(), **locals()))

def dev_resave_pngs(input_dir):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('input_dir not found. Please ensure it exists.')

    images_paths = pathex.get_image_paths(input_path, image_extensions=['.png'], subdirs=True, return_Path_class=True)

    for filepath in io.progress_bar_generator(images_paths,"Processing"):
        cv2_imwrite(filepath, cv2_imread(filepath))


def dev_segmented_trash(input_dir):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('input_dir not found. Please ensure it exists.')

    output_path = input_path.parent / (input_path.name+'_trash')
    output_path.mkdir(parents=True, exist_ok=True)

    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)

    trash_paths = []
    for filepath in images_paths:
        json_file = filepath.parent / (filepath.stem +'.json')
        if not json_file.exists():
            trash_paths.append(filepath)

    for filepath in trash_paths:

        try:
            filepath.rename ( output_path / filepath.name )
        except:
            io.log_info ('fail to trashing %s' % (src.name) )

    
def dev_segmented_extract(input_dir, output_dir ):
    # extract and merge .json labelme files within the faces

    device_config = nn.DeviceConfig.GPUIndexes( nn.ask_choose_device_idxs(suggest_all_gpu=True) )

    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('input_dir not found. Please ensure it exists.')

    output_path = Path(output_dir)
    io.log_info("Performing extract segmented faces.")
    io.log_info(f'Output dir is {output_path}')

    if output_path.exists():
        output_images_paths = pathex.get_image_paths(output_path, subdirs=True)
        if len(output_images_paths) > 0:
            io.input_bool("WARNING !!! \n %s contains files! \n They will be deleted. \n Press enter to continue." % (str(output_path)), False )
            for filename in output_images_paths:
                Path(filename).unlink()
            shutil.rmtree(str(output_path))
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    images_paths = pathex.get_image_paths(input_path, subdirs=True, return_Path_class=True)

    extract_data = []
    images_jsons = {}
    images_processed = 0


    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        json_filepath = filepath.parent / (filepath.stem+'.json')

        if json_filepath.exists():
            try:
                json_dict = json.loads(json_filepath.read_text())
                images_jsons[filepath] = json_dict

                total_points = [ [x,y] for shape in json_dict['shapes'] for x,y in shape['points'] ]
                total_points = np.array(total_points)

                if len(total_points) == 0:
                    io.log_info(f"No points found in {json_filepath}, skipping.")
                    continue

                l,r = int(total_points[:,0].min()), int(total_points[:,0].max())
                t,b = int(total_points[:,1].min()), int(total_points[:,1].max())

                force_output_path=output_path / filepath.relative_to(input_path).parent
                force_output_path.mkdir(exist_ok=True, parents=True)

                extract_data.append ( ExtractSubprocessor.Data(filepath,
                                                               rects=[ [l,t,r,b] ],
                                                               force_output_path=force_output_path ) )
                images_processed += 1
            except:
                io.log_err(f"err {filepath}, {traceback.format_exc()}")
                return
        else:
            io.log_info(f"No .json file for {filepath.relative_to(input_path)}, skipping.")
            continue

    image_size = 1024
    face_type = FaceType.HEAD
    extract_data = ExtractSubprocessor (extract_data, 'landmarks', image_size, face_type, device_config=device_config).run()
    extract_data = ExtractSubprocessor (extract_data, 'final', image_size, face_type, device_config=device_config).run()

    for data in extract_data:
        filepath = data.force_output_path / (data.filepath.stem+'_0.jpg')

        dflimg = DFLIMG.load(filepath)
        image_to_face_mat = dflimg.get_image_to_face_mat()

        json_dict = images_jsons[data.filepath]

        ie_polys = IEPolys()
        for shape in json_dict['shapes']:
            ie_poly = ie_polys.add(1)

            points = np.array( [ [x,y] for x,y in shape['points'] ] )
            points = LandmarksProcessor.transform_points(points, image_to_face_mat)

            for x,y in points:
                ie_poly.add( int(x), int(y) )

        dflimg.embed_and_set (filepath, ie_polys=ie_polys)

    io.log_info(f"Images found:     {len(images_paths)}")
    io.log_info(f"Images processed: {images_processed}")



"""
#mark only
for data in extract_data:
    filepath = data.filepath
    output_filepath = output_path / (filepath.stem+'.jpg')

    img = cv2_imread(filepath)
    img = imagelib.normalize_channels(img, 3)
    cv2_imwrite(output_filepath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100] )

    json_dict = images_jsons[filepath]

    ie_polys = IEPolys()
    for shape in json_dict['shapes']:
        ie_poly = ie_polys.add(1)
        for x,y in shape['points']:
            ie_poly.add( int(x), int(y) )


    DFLJPG.embed_data(output_filepath, face_type=FaceType.toString(FaceType.MARK_ONLY),
                                        landmarks=data.landmarks[0],
                                        ie_polys=ie_polys,
                                        source_filename=filepath.name,
                                        source_rect=data.rects[0],
                                        source_landmarks=data.landmarks[0]
                        )
"""