from enum import IntEnum

class FaceType(IntEnum):
    #enumerating in order "next contains prev"
    HALF = 0
    MID_FULL = 1
    FULL = 2
    FULL_NO_ALIGN = 3
    HEAD = 4
    HEAD_NO_ALIGN = 5
        
    MARK_ONLY = 10, #no align at all, just embedded faceinfo

    @staticmethod
    def fromString (s):
        r = from_string_dict.get (s.lower())
        if r is None:
            raise Exception ('FaceType.fromString value error')
        return r

    @staticmethod
    def toString (face_type):
        return to_string_dict[face_type]

from_string_dict = {'half_face': FaceType.HALF,
                    'midfull_face': FaceType.MID_FULL,
                    'full_face': FaceType.FULL,
                    'head' : FaceType.HEAD,
                    'mark_only' : FaceType.MARK_ONLY,
                    'full_face_no_align' : FaceType.FULL_NO_ALIGN,
                    'head_no_align' : FaceType.HEAD_NO_ALIGN,
                    }
to_string_dict = { FaceType.HALF : 'half_face',
                   FaceType.MID_FULL : 'midfull_face',
                   FaceType.FULL : 'full_face',
                   FaceType.HEAD : 'head',
                   FaceType.MARK_ONLY :'mark_only',
                   FaceType.FULL_NO_ALIGN : 'full_face_no_align',
                   FaceType.HEAD_NO_ALIGN : 'head_no_align'
                 }
