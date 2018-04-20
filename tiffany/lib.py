import ctypes

import numpy as np

from . import config

_LIB = config.load_library() 

_TAGS = {
    'width': {
        'number': 256,
        'type': ctypes.c_uint32,
    },
    'length': {
        'number': 257,
        'type': ctypes.c_uint32,
    },
    'bitspersample': {
        'number': 258,
        'type': ctypes.c_uint16,
    },
    'sampleformat': {
        'number': 339,
        'type': ctypes.c_uint16,
    },
}

ORIENTATION_TOPLEFT = 1	# row 0 top, col 0 lhs */
ORIENTATION_TOPRIGHT = 2	# row 0 top, col 0 rhs */
ORIENTATION_BOTRIGHT = 3	# row 0 bottom, col 0 rhs */
ORIENTATION_BOTLEFT	= 4	# row 0 bottom, col 0 lhs */
ORIENTATION_LEFTTOP	= 5	# row 0 lhs, col 0 top */
ORIENTATION_RIGHTTOP = 6	# row 0 rhs, col 0 top */
ORIENTATION_RIGHTBOT = 7	# row 0 rhs, col 0 bottom */
ORIENTATION_LEFTBOT	= 8	# row 0 lhs, col 0 bottom */

def open(filename, mode='r'):
    """
    Corresponds to TIFFOpen
    """
    ARGTYPES = [ctypes.c_char_p, ctypes.c_char_p]
    _LIB.TIFFOpen.argtypes = ARGTYPES
    _LIB.TIFFOpen.restype = ctypes.c_void_p
    file_argument = ctypes.c_char_p(filename.encode())
    mode_argument = ctypes.c_char_p(mode.encode())
    fp = _LIB.TIFFOpen(file_argument, mode_argument)
    return fp


def getField(fp, tag):
    ARGTYPES = [ctypes.c_void_p, ctypes.c_int32]

    tag_num = _TAGS[tag]['number']

    # Append the proper return type for the tag.
    tag_type = _TAGS[tag]['type']
    ARGTYPES.append(ctypes.POINTER(_TAGS[tag]['type']))
    _LIB.TIFFGetField.argtypes = ARGTYPES

    _LIB.TIFFGetField.restype = ctypes.c_int

    # instantiate the tag value
    item = tag_type()
    status = _LIB.TIFFGetField(fp, tag_num, ctypes.byref(item))
    return item.value

def getFieldDefaulted(fp, tag):
    ARGTYPES = [ctypes.c_void_p, ctypes.c_int32]

    tag_num = _TAGS[tag]['number']

    # Append the proper return type for the tag.
    tag_type = _TAGS[tag]['type']
    ARGTYPES.append(ctypes.POINTER(_TAGS[tag]['type']))
    _LIB.TIFFGetFieldDefaulted.argtypes = ARGTYPES

    _LIB.TIFFGetFieldDefaulted.restype = ctypes.c_int

    # instantiate the tag value
    item = tag_type()
    status = _LIB.TIFFGetFieldDefaulted(fp, tag_num, ctypes.byref(item))
    return item.value

def readRGBAImage(fp, width=None, height=None,
                  orientation=ORIENTATION_TOPLEFT, stopOnError=0):
    """
    Corresponds to TIFFReadRGBAImage

    int  TIFFReadRGBAImageOriented(
        TIFF  *tif,
        uint32 width, uint32 height,
        uint32 *raster,
        int orientation, int stopOnError
    )
    """
    ARGTYPES = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int32, ctypes.c_int32
    ]

    _LIB.TIFFReadRGBAImage.argtypes = ARGTYPES
    _LIB.TIFFReadRGBAImage.restype = ctypes.c_int

    img = np.zeros((height, width, 4), dtype=np.uint8)
    raster = img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    status = _LIB.TIFFReadRGBAImage(fp, width, height, raster,
                                    3, stopOnError)
    return img
