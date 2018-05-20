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
    'stripoffsets': {
        'number': 273,
        'type': (ctypes.c_uint32, ctypes.c_uint64),
    },
    'stripbytecounts': {
        'number': 279,
        'type': None,
    },
    'planarconfig': {
        'number': 284,
        'type': ctypes.c_uint16,
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
    'samplesperpixel': {
        'number': 277,
        'type': ctypes.c_uint16,
    },
    'rowsperstrip': {
        'number': 278,
        'type': ctypes.c_uint16,
    },
    'tilewidth': {
        'number': 322,
        'type': ctypes.c_uint32,
    },
    'tilelength': {
        'number': 323,
        'type': ctypes.c_uint32,
    },
}

# We need the reverse mapping as well.
tagnum2name = {value['number']: key for key, value in TAGS.items()}
