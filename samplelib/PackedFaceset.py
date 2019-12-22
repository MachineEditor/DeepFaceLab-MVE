import pickle
import struct
from pathlib import Path

from interact import interact as io
from utils import Path_utils


import samplelib.SampleHost
from samplelib import Sample

packed_faceset_filename = 'faceset.pak'

class PackedFaceset():
    VERSION = 1

    @staticmethod
    def pack(samples_path):
        samples_dat_path = samples_path / packed_faceset_filename

        if samples_dat_path.exists():
            io.log_info(f"{samples_dat_path} : file already exists !")
            io.input_bool("Press enter to continue and overwrite.", False)

        of = open(samples_dat_path, "wb")

        image_paths = Path_utils.get_image_paths(samples_path)


        samples = samplelib.SampleHost.load_face_samples(image_paths)
        samples_len = len(samples)

        samples_configs = []
        for sample in samples:
            sample.filename = str(Path(sample.filename).relative_to(samples_path))
            samples_configs.append ( sample.get_config() )
        samples_bytes = pickle.dumps(samples_configs, 4)

        of.write ( struct.pack ("Q", PackedFaceset.VERSION ) )
        of.write ( struct.pack ("Q", len(samples_bytes) ) )
        of.write ( samples_bytes )

        sample_data_table_offset = of.tell()
        of.write ( bytes( 8*(samples_len+1) ) ) #sample data offset table

        data_start_offset = of.tell()
        offsets = []

        for sample in io.progress_bar_generator(samples, "Packing"):
            try:
                with open( samples_path / sample.filename, "rb") as f:
                    b = f.read()

                offsets.append ( of.tell() - data_start_offset )
                of.write(b)
            except:
                raise Exception(f"error while processing sample {sample.filename}")

        offsets.append ( of.tell() )

        of.seek(sample_data_table_offset, 0)
        for offset in offsets:
            of.write ( struct.pack("Q", offset) )
        of.seek(0,2)
        of.close()

        for filename in io.progress_bar_generator(image_paths,"Deleting"):
            Path(filename).unlink()


    @staticmethod
    def unpack(samples_path):
        samples_dat_path = samples_path / packed_faceset_filename
        if not samples_dat_path.exists():
            io.log_info(f"{samples_dat_path} : file not found.")
            return

        samples = PackedFaceset.load(samples_path)

        for sample in io.progress_bar_generator(samples, "Unpacking"):
            with open(samples_path / sample.filename, "wb") as f:
                f.write( sample.read_raw_file() )

        samples_dat_path.unlink()

    @staticmethod
    def load(samples_path):
        samples_dat_path = samples_path / packed_faceset_filename
        if not samples_dat_path.exists():
            return None

        f = open(samples_dat_path, "rb")
        version, = struct.unpack("Q", f.read(8) )
        if version != PackedFaceset.VERSION:
            raise NotImplementedError

        sizeof_samples_bytes, = struct.unpack("Q", f.read(8) )

        samples_configs = pickle.loads ( f.read(sizeof_samples_bytes) )
        samples = []
        for sample_config in samples_configs:
            samples.append ( Sample (**sample_config) )        
        
        offsets = [ struct.unpack("Q", f.read(8) )[0] for _ in range(len(samples)+1) ]
        data_start_offset = f.tell()
        f.close()

        for i, sample in enumerate(samples):
            start_offset, end_offset = offsets[i], offsets[i+1]
            sample.set_filename_offset_size( str(samples_dat_path), data_start_offset+start_offset, end_offset-start_offset )

        return samples

