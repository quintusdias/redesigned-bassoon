# Standard library imports ...
import ctypes

TAGS = {
    'subfiletype': {
        'number': 255,
        'type': ctypes.c_uint16,
    },
    'imagewidth': {
        'number': 256,
        'type': ctypes.c_uint32,
    },
    'imagelength': {
        'number': 257,
        'type': ctypes.c_uint32,
    },
    'bitspersample': {
        'number': 258,
        'type': ctypes.c_uint16,
    },
    'compression': {
        'number': 259,
        'type': ctypes.c_uint16,
    },
    'photometric': {
        'number': 262,
        'type': ctypes.c_uint16,
    },
    'fillorder': {
        'number': 266,
        'type': ctypes.c_uint16,
    },
    'documentname': {
        'number': 269,
        'type': ctypes.c_char,
    },
    'imagedescription': {
        'number': 270,
        'type': ctypes.c_char,
    },
    'stripoffsets': {
        'number': 273,
        'type': (ctypes.c_uint32, ctypes.c_uint64),
    },
    'orientation': {
        'number': 274,
        'type': ctypes.c_uint16,
    },
    'samplesperpixel': {
        'number': 277,
        'type': ctypes.c_uint16,
    },
    'rowsperstrip': {
        'number': 278,
        'type': ctypes.c_uint16,
    },
    'stripbytecounts': {
        'number': 279,
        'type': None,
    },
    'minsamplevalue': {
        'number': 280,
        'type': ctypes.c_uint16,
    },
    'maxsamplevalue': {
        'number': 281,
        'type': ctypes.c_uint16,
    },
    'xresolution': {
        'number': 282,
        'type': ctypes.c_double,
    },
    'yresolution': {
        'number': 283,
        'type': ctypes.c_double,
    },
    'planarconfig': {
        'number': 284,
        'type': ctypes.c_uint16,
    },
    't4options': {
        'number': 292,
        'type': None,
    },
    'resolutionunit': {
        'number': 296,
        'type': ctypes.c_uint16,
    },
    'pagenumber': {
        'number': 297,
        'type': (ctypes.c_uint16, ctypes.c_uint16),
    },
    'tilewidth': {
        'number': 322,
        'type': ctypes.c_uint32,
    },
    'software': {
        'number': 305,
        'type': ctypes.c_char,
    },
    'colormap': {
        'number': 320,
        'type': (ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16),
    },
    'tilelength': {
        'number': 323,
        'type': ctypes.c_uint32,
    },
    'tileoffsets': {
        'number': 324,
        'type': None,
    },
    'tilebytecounts': {
        'number': 325,
        'type': None,
    },
    'badfaxlines': {
        'number': 326,
        'type': None,
    },
    'cleanfaxdata': {
        'number': 327,
        'type': None,
    },
    'consecutivebadfaxlines': {
        'number': 328,
        'type': None,
    },
    'sampleformat': {
        'number': 339,
        'type': ctypes.c_uint16,
    },
    'sminsamplevalue': {
        'number': 340,
        'type': ctypes.c_double,
    },
    'smaxsamplevalue': {
        'number': 341,
        'type': ctypes.c_double,
    },
    'jpegtables': {
        'number': 347,
        'type': None,
    },
    'jpegproc': {
        'number': 512,
        'type': None,
    },
    'jpegqtables': {
        'number': 519,
        'type': None,
    },
    'jpegdctables': {
        'number': 520,
        'type': None,
    },
    'jpegactables': {
        'number': 521,
        'type': None,
    },
    'ycbcrcoefficients': {
        'number': 529,
        'type': (ctypes.c_float, ctypes.c_float, ctypes.c_float),
    },
    'ycbcrsubsampling': {
        'number': 530,
        'type': (ctypes.c_uint16, ctypes.c_uint16),
    },
    'referenceblackwhite': {
        'number': 532,
        'type': (ctypes.c_float, ctypes.c_float, ctypes.c_float,
                 ctypes.c_float, ctypes.c_float, ctypes.c_float),
    },
    'datatype': {
        'number': 32996,
        'type': None,
    },
    'imagedepth': {
        'number': 32997,
        'type': None,
    },
    'tiledepth': {
        'number': 32998,
        'type': None,
    },
    'jpegcolormode': {
        'number': 65538,
        'type': ctypes.c_int32,
    },
}

# We need the reverse mapping as well.
tagnum2name = {value['number']: key for key, value in TAGS.items()}
