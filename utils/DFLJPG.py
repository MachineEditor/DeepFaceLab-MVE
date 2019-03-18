import struct
import pickle
import numpy as np
from facelib import FaceType
from utils.struct_utils import *

class DFLJPG(object):
    def __init__(self):
        self.data = b""
        self.length = 0
        self.chunks = []
        self.dfl_dict = None
        self.shape = (0,0,0)
        
    @staticmethod
    def load_raw(filename):
        try:
            with open(filename, "rb") as f:
                data = f.read()
        except:
            raise FileNotFoundError(data)
    
        try:
            inst = DFLJPG()
            inst.data = data
            inst.length = len(data)
            inst_length = inst.length
            chunks = []
            data_counter = 0
            while data_counter < inst_length:
                chunk_m_l, chunk_m_h = struct.unpack ("BB", data[data_counter:data_counter+2])
                data_counter += 2
                
                if chunk_m_l != 0xFF:
                    raise ValueError("No Valid JPG info")
                
                chunk_name = None
                chunk_size = None
                chunk_data = None
                chunk_ex_data = None
                is_unk_chunk = False
                
                if chunk_m_h & 0xF0 == 0xD0:            
                    n = chunk_m_h & 0x0F
                    
                    if n >= 0 and n <= 7: 
                        chunk_name = "RST%d" % (n)
                        chunk_size = 0
                    elif n == 0x8: 
                        chunk_name = "SOI"
                        chunk_size = 0
                        if len(chunks) != 0:
                            raise Exception("")
                    elif n == 0x9:
                        chunk_name = "EOI"
                        chunk_size = 0
                    elif n == 0xA: 
                        chunk_name = "SOS"                        
                    elif n == 0xB: 
                        chunk_name = "DQT"
                    elif n == 0xD:
                        chunk_name = "DRI"
                        chunk_size = 2
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xC0:            
                    n = chunk_m_h & 0x0F            
                    if n == 0: 
                        chunk_name = "SOF0"
                    elif n == 2: 
                        chunk_name = "SOF2"
                    elif n == 4: 
                        chunk_name = "DHT"
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xE0:            
                    n = chunk_m_h & 0x0F
                    chunk_name = "APP%d" % (n)
                else:
                    is_unk_chunk = True
                    
                if is_unk_chunk:
                    raise ValueError("Unknown chunk %X" % (chunk_m_h) )    
        
                if chunk_size == None: #variable size
                    chunk_size, = struct.unpack (">H", data[data_counter:data_counter+2])
                    chunk_size -= 2
                    data_counter += 2
                
                if chunk_size > 0:
                    chunk_data = data[data_counter:data_counter+chunk_size]
                    data_counter += chunk_size
                
                if chunk_name == "SOS":
                    c = data_counter                        
                    while c < inst_length and (data[c] != 0xFF or data[c+1] != 0xD9):
                        c += 1
                        
                    chunk_ex_data = data[data_counter:c]
                    data_counter = c
                
                chunks.append ({'name' : chunk_name,
                                'm_h' : chunk_m_h,
                                'data' : chunk_data,
                                'ex_data' : chunk_ex_data,
                                })                                
            inst.chunks = chunks
            
            return inst
        except Exception as e:
            raise Exception ("Corrupted JPG file: %s" % (str(e)))
 
    @staticmethod
    def load(filename):
        try:
            inst = DFLJPG.load_raw (filename)
            inst.dfl_dict = None
            
            for chunk in inst.chunks:
                if chunk['name'] == 'APP0':
                    d, c = chunk['data'], 0
                    c, id, _ = struct_unpack (d, c, "=4sB")
                    
                    if id == b"JFIF":
                        c, ver_major, ver_minor, units, Xdensity, Ydensity, Xthumbnail, Ythumbnail = struct_unpack (d, c, "=BBBHHBB")
                        #if units == 0:
                        #    inst.shape = (Ydensity, Xdensity, 3)
                    else:
                        raise Exception("Unknown jpeg ID: %s" % (id) )
                elif chunk['name'] == 'SOF0' or chunk['name'] == 'SOF2':
                    d, c = chunk['data'], 0
                    c, precision, height, width = struct_unpack (d, c, ">BHH")
                    inst.shape = (height, width, 3)
                    
                elif chunk['name'] == 'APP15':
                    if type(chunk['data']) == bytes:
                        inst.dfl_dict = pickle.loads(chunk['data'])

            if (inst.dfl_dict is not None) and ('face_type' not in inst.dfl_dict.keys()):
                inst.dfl_dict['face_type'] = FaceType.toString (FaceType.FULL)
            
            if inst.dfl_dict == None:
                return None
            
            return inst
        except Exception as e:
            print (e)
            return None
        
    @staticmethod
    def embed_data(filename, face_type=None,
                             landmarks=None,
                             source_filename=None,
                             source_rect=None,
                             source_landmarks=None,
                             image_to_face_mat=None
                   ):
    
        inst = DFLJPG.load_raw (filename)
        inst.setDFLDictData ({
                                'face_type': face_type,
                                'landmarks': landmarks,
                                'source_filename': source_filename,
                                'source_rect': source_rect,
                                'source_landmarks': source_landmarks,
                                'image_to_face_mat': image_to_face_mat
                             })
    
        try:
            with open(filename, "wb") as f:
                f.write ( inst.dump() )
        except:
            raise Exception( 'cannot save %s' % (filename) )
            
    def dump(self):
        data = b""
        
        for chunk in self.chunks:
            data += struct.pack ("BB", 0xFF, chunk['m_h'] )
            chunk_data = chunk['data']
            if chunk_data is not None:
                data += struct.pack (">H", len(chunk_data)+2 )
                data += chunk_data
                
            chunk_ex_data = chunk['ex_data']
            if chunk_ex_data is not None:      
                data += chunk_ex_data

        return data
        
    def get_shape(self):        
        return self.shape
        
    def get_height(self):
        for chunk in self.chunks:
            if type(chunk) == IHDR:
                return chunk.height
        return 0
        
    def getDFLDictData(self):
        return self.dfl_dict
                
    def setDFLDictData (self, dict_data=None):
        self.dfl_dict = dict_data

        for chunk in self.chunks:
            if chunk['name'] == 'APP15':
                self.chunks.remove(chunk)
                break

        last_app_chunk = 0
        for i, chunk in enumerate (self.chunks):
            if chunk['m_h'] & 0xF0 == 0xE0:
                last_app_chunk = i            
        
        dflchunk = {'name' : 'APP15',
                    'm_h' : 0xEF,
                    'data' : pickle.dumps(dict_data),
                    'ex_data' : None,
                    }
        self.chunks.insert (last_app_chunk+1, dflchunk)
       
    def get_face_type(self): return self.dfl_dict['face_type']
    def get_landmarks(self): return np.array ( self.dfl_dict['landmarks'] )   
    def get_source_filename(self): return self.dfl_dict['source_filename']        
    def get_source_rect(self): return self.dfl_dict['source_rect']        
    def get_source_landmarks(self): return np.array ( self.dfl_dict['source_landmarks'] )
