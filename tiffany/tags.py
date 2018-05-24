# Standard library imports ...
import ctypes

TAGS = {
    'bitspersample': {
        'number': 258,
        'type': ctypes.c_uint16,
    },
    'compression': {
        'number': 259,
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
    'photometric': {
        'number': 262,
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
    'sampleformat': {
        'number': 339,
        'type': ctypes.c_uint16,
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
    'jpegcolormode': {
        'number': 65538,
        'type': ctypes.c_int32,
    },
}

# We need the reverse mapping as well.
tagnum2name = {value['number']: key for key, value in TAGS.items()}
