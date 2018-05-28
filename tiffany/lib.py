# Standard library imports
import ctypes
import datetime
from enum import IntEnum

# Third party library imports
import numpy as np

# Local imports
from . import config
from .tags import TAGS

_LIB = config.load_library()


class NotRGBACompatibleError(RuntimeError):
    """
    Raise this exception if an attempt is made to set the rgba property on an
    incompatible image.

    See TIFFRGBAImage(3).
    """
    pass


class Compression(IntEnum):
    """
    Corresponds to COMPRESSION_* values listed in tiff.h
    """
    NONE = 1
    CCITTRLE = 2  # CCITT modified Huffman RLE
    CCITTFAX3 = 3  # CCITT Group 3 fax encoding
    CCITT_T4 = 3  # CCITT T.4 (TIFF 6 name)
    CCITTFAX4 = 4  # CCITT Group 4 fax encoding
    CCITT_T6 = 4  # CCITT T.6 (TIFF 6 name)
    LZW = 5  # Lempel-Ziv  & Welch
    OJPEG = 6  # 6.0 JPEG
    JPEG = 7  # JPEG DCT compression
    T85 = 9  # TIFF/FX T.85 JBIG compression
    T43 = 10  # TIFF/FX T.43 colour by layered JBIG compression
    NEXT = 32766  # NeXT 2-bit RLE
    CCITTRLEW = 32771  # #1 w/ word alignment
    PACKBITS = 32773  # Macintosh RLE
    THUNDERSCAN = 32809  # ThunderScan RLE
    PIXARFILM = 32908   # companded 10bit LZW
    PIXARLOG = 32909   # companded 11bit ZIP
    DEFLATE = 32946  # compression
    ADOBE_DEFLATE = 8       # compression, as recognized by Adobe
    DCS = 32947   # DCS encoding
    JBIG = 34661  # JBIG
    SGILOG = 34676  # Log Luminance RLE
    SGILOG24 = 34677  # Log 24-bit packed
    JP2000 = 34712   # JPEG2000
    LZMA = 34925  # LZMA2


class ExtraSamples(IntEnum):
    """
    Corresponds to EXTRASAM* values listed in tiff.h
    """
    UNSPECIFIED = 0
    ASSOCALPHA = 1
    UNASSALPHA = 2


class FillOrder(IntEnum):
    """
    Corresponds to TIFFTAG_FILLORDER* values listed in tiff.h
    """
    MSB2LSB = 1  # most significant -> least
    LSB2MSB = 2  # least significant -> most


class InkSet(IntEnum):
    """
    Corresponds to INKSET** values listed in tiff.h
    """
    CMYK = 1
    MULTIINK = 2


class Orientation(IntEnum):
    """
    Corresponds to ORIENTATION_* values listed in tiff.h
    """
    TOPLEFT = 1  # row 0 top, col 0 lhs */
    TOPRIGHT = 2  # row 0 top, col 0 rhs */
    BOTRIGHT = 3  # row 0 bottom, col 0 rhs */
    BOTLEFT = 4  # row 0 bottom, col 0 lhs */
    LEFTTOP = 5  # row 0 lhs, col 0 top */
    RIGHTTOP = 6  # row 0 rhs, col 0 top */
    RIGHTBOT = 7  # row 0 rhs, col 0 bottom */
    LEFTBOT = 8  # row 0 lhs, col 0 bottom */


class Photometric(IntEnum):
    """
    Corresponds to TIFFTAG_PHOTOMETRIC* values listed in tiff.h
    """
    MINISWHITE = 0  # value is white
    MINISBLACK = 1  # value is black
    RGB = 2  # color model
    PALETTE = 3  # map indexed
    MASK = 4  # holdout mask
    SEPARATED = 5  # color separations
    YCBCR = 6  # CCIR 601
    CIELAB = 8  # 1976 CIE L*a*b*
    ICCLAB = 9  # L*a*b* [Adobe TIFF Technote 4]
    ITULAB = 10  # L*a*b*
    CFA = 32803  # filter array
    LOGL = 32844  # Log2(L)
    LOGLUV = 32845  # Log2(L) (u',v')


class JPEGColorMode(IntEnum):
    """
    Corresponds to TIFFTAG_JPEGCOLORMODE values listed in tiff.h
    """
    RAW = 0
    RGB = 1


class JPEGProc(IntEnum):
    """
    Corresponds to JPEGPROC* values listed in tiff.h
    """
    BASELINE = 1
    LOSSLESS = 14


class OSubFileType(IntEnum):
    """
    Corresponds to OFILETYPE* values listed in tiff.h
    """
    IMAGE = 1
    REDUCEDIMAGE = 2
    PAGE = 3


class PlanarConfig(IntEnum):
    """
    Corresponds to TIFFTAG_PLANARCONFIG* values listed in tiff.h
    """
    CONTIG = 1  # single image plane
    SEPARATE = 2  # separate planes of data


class Predictor(IntEnum):
    """
    Corresponds to PREDICTOR* values listed in tiff.h
    """
    NONE = 1
    HORIZONTAL = 2
    FLOATINGPOINT = 3


class ResolutionUnit(IntEnum):
    """
    Corresponds to RESUNIT* values listed in tiff.h
    """
    NONE = 1
    INCH = 2
    CENTIMETER = 3


class SampleFormat(IntEnum):
    """
    Corresponds to values listed in tiff.h
    """
    UINT = 1
    INT = 2
    IEEEFP = 3
    VOID = 4
    COMPLEXINT = 5
    COMPLEXIEEEP = 6


class SubFileType(IntEnum):
    """
    Corresponds to FILETYPE* values listed in tiff.h
    """
    REDUCEDIMAGE = 1
    PAGE = 2
    MASK = 4


class THRESHHOLDING(IntEnum):
    """
    Corresponds to THRESHHOLD* values listed in tiff.h
    """
    BILEVEL = 1
    HALFTONE = 2
    ERRORDIFFUSE = 3


class T4Options(IntEnum):
    """
    Corresponds to GROUP3OPT* values listed in tiff.h
    """
    TWOD_ENCODING = 1
    UNCOMPRESSED = 2
    FILLBITS = 4


class YCbCrPosition(IntEnum):
    """
    Corresponds to YCBCRPOSITION* values listed in tiff.h
    """
    CENTERED = 1
    COSITED = 2


def close(fp):
    """
    Corresponds to TIFFClose
    """
    ARGTYPES = [ctypes.c_void_p]
    _LIB.TIFFClose.argtypes = ARGTYPES
    _LIB.TIFFClose.restype = None
    _LIB.TIFFClose(fp)


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


def setField(fp, tag, value):
    """
    Corresponds to TIFFSetField
    """
    ARGTYPES = [ctypes.c_void_p, ctypes.c_int32]

    # Append the proper return type for the tag.
    tag_num = TAGS[tag]['number']
    tag_type = TAGS[tag]['type']
    if tag_num == 530:
        ARGTYPES.extend(tag_type)
    else:
        ARGTYPES.append(tag_type)
    _LIB.TIFFSetField.argtypes = ARGTYPES
    _LIB.TIFFSetField.restype = check_error

    if tag_num == 530:
        _LIB.TIFFSetField(fp, tag_num, value[0], value[1])
    elif tag_num == 306 and isinstance(value, datetime.datetime):
        value = value.strftime('%Y:%m:%d %H:%M:%S').encode('utf-8')
        _LIB.TIFFSetField(fp, tag_num, ctypes.c_char_p(value))
    elif tag_type == ctypes.c_char_p:
        value = value.encode('utf-8')
        _LIB.TIFFSetField(fp, tag_num, ctypes.c_char_p(value))
    else:
        _LIB.TIFFSetField(fp, tag_num, value)


def computeStrip(fp, row, sample):
    """
    Corresponds to TIFFComputeStrip
    """
    ARGTYPES = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint16]
    _LIB.TIFFComputeStrip.argtypes = ARGTYPES
    _LIB.TIFFComputeStrip.restype = ctypes.c_uint32
    stripnum = _LIB.TIFFComputeStrip(fp, row, sample)
    return stripnum


def computeTile(fp, x, y, sample):
    """
    Corresponds to TIFFComputeTile
    """
    ARGTYPES = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint16]
    _LIB.TIFFComputeTile.argtypes = ARGTYPES
    _LIB.TIFFComputeTile.restype = ctypes.c_uint32
    tilenum = _LIB.TIFFComputeTile(fp, x, y, sample)
    return tilenum


def readEncodedStrip(fp, stripnum):
    """
    Corresponds to TIFFReadEncodedStrip
    """
    ARGTYPES = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32
    ]
    _LIB.TIFFReadEncodedStrip.argtypes = ARGTYPES
    _LIB.TIFFReadEncodedStrip.restype = check_error
    rps = getField(fp, 'RowsPerStrip')
    width = getField(fp, 'ImageWidth')
    spp = getField(fp, 'SamplesPerPixel')
    shape = (rps, width, spp)
    image = np.zeros(shape, dtype=np.uint8)
    _LIB.TIFFReadEncodedStrip(fp, stripnum,
                              image.ctypes.data_as(ctypes.c_void_p), -1)
    return image


def readEncodedTile(fp, tilenum):
    """
    Corresponds to TIFFComputeTile
    """
    ARGTYPES = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                ctypes.c_int32]
    _LIB.TIFFReadEncodedTile.argtypes = ARGTYPES
    _LIB.TIFFReadEncodedTile.restype = check_error
    tilelength = getField(fp, 'TileLength')
    tilewidth = getField(fp, 'TileWidth')
    spp = getField(fp, 'SamplesPerPixel')
    shape = (tilelength, tilewidth, spp)
    image = np.zeros(shape, dtype=np.uint8)
    _LIB.TIFFReadEncodedTile(fp, tilenum,
                             image.ctypes.data_as(ctypes.c_void_p), -1)
    return image


def getField(fp, tag):
    ARGTYPES = [ctypes.c_void_p, ctypes.c_int32]

    tag_num = TAGS[tag]['number']

    # Append the proper return type for the tag.
    tag_type = TAGS[tag]['type']
    ARGTYPES.append(ctypes.POINTER(tag_type))
    _LIB.TIFFGetField.argtypes = ARGTYPES

    _LIB.TIFFGetField.restype = check_error

    # instantiate the tag value
    item = tag_type()
    _LIB.TIFFGetField(fp, tag_num, ctypes.byref(item))
    return item.value


def getFieldDefaulted(fp, tag):
    ARGTYPES = [ctypes.c_void_p, ctypes.c_int32]

    tag_num = TAGS[tag]['number']

    # Append the proper return type for the tag.
    tag_type = TAGS[tag]['type']
    ARGTYPES.append(ctypes.POINTER(TAGS[tag]['type']))
    _LIB.TIFFGetFieldDefaulted.argtypes = ARGTYPES

    _LIB.TIFFGetFieldDefaulted.restype = check_error

    # instantiate the tag value
    item = tag_type()
    _LIB.TIFFGetFieldDefaulted(fp, tag_num, ctypes.byref(item))
    return item.value


def isTiled(fp):
    """
    Corresponds to TIFFIsTiled
    """
    ARGTYPES = [ctypes.c_void_p]

    _LIB.TIFFIsTiled.argtypes = ARGTYPES
    _LIB.TIFFIsTiled.restype = ctypes.c_int

    status = _LIB.TIFFIsTiled(fp)
    return status


def numberOfStrips(fp):
    """
    Corresponds to TIFFNumberOfStrips.
    """
    ARGTYPES = [ctypes.c_void_p]
    _LIB.TIFFNumberOfStrips.argtypes = ARGTYPES
    _LIB.TIFFNumberOfStrips.restype = ctypes.c_uint32

    n = _LIB.TIFFNumberOfStrips(fp)
    return n


def numberOfTiles(fp):
    """
    Corresponds to TIFFNumberOfTiles.
    """
    ARGTYPES = [ctypes.c_void_p]
    _LIB.TIFFNumberOfTiles.argtypes = ARGTYPES
    _LIB.TIFFNumberOfTiles.restype = ctypes.c_uint32

    numtiles = _LIB.TIFFNumberOfTiles(fp)
    return numtiles


def readRGBAImageOriented(fp, width=None, height=None,
                          orientation=Orientation.TOPLEFT, stopOnError=0):
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

    _LIB.TIFFReadRGBAImageOriented.argtypes = ARGTYPES
    _LIB.TIFFReadRGBAImageOriented.restype = check_error

    img = np.zeros((height, width, 4), dtype=np.uint8)
    raster = img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    _LIB.TIFFReadRGBAImageOriented(fp, width, height, raster, orientation,
                                   stopOnError)
    return img


def RGBAImageOK(fp):
    """
    Corresponds to TIFFRGBAImageOK.
    """
    emsg = ctypes.create_string_buffer(1024)
    ARGTYPES = [ctypes.c_void_p, ctypes.c_char_p]
    _LIB.TIFFRGBAImageOK.argtypes = ARGTYPES
    _LIB.TIFFRGBAImageOK.restype = ctypes.c_int
    ok = _LIB.TIFFRGBAImageOK(fp, emsg)
    if not ok:
        error_message = f"libtiff:  {emsg.value.decode('utf-8')}"
        raise NotRGBACompatibleError(error_message)


def writeEncodedStrip(fp, stripnum, stripdata, size=-1):
    """
    Corresponds to TIFFWriteEncodedStrip.
    """
    ARGTYPES = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32
    ]
    _LIB.TIFFWriteEncodedStrip.argtypes = ARGTYPES
    _LIB.TIFFWriteEncodedStrip.restype = ctypes.c_int
    raster = stripdata.ctypes.data_as(ctypes.c_void_p)
    _LIB.TIFFWriteEncodedStrip(fp, stripnum, raster, size)


def writeEncodedTile(fp, tilenum, tiledata, size=-1):
    """
    Corresponds to TIFFWriteEncodedTile.
    """
    ARGTYPES = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32
    ]
    _LIB.TIFFWriteEncodedTile.argtypes = ARGTYPES
    _LIB.TIFFWriteEncodedTile.restype = check_error
    raster = tiledata.ctypes.data_as(ctypes.c_void_p)
    _LIB.TIFFWriteEncodedTile(fp, tilenum, raster, size)


def check_error(status):
    """
    Set a generic function as the restype attribute of all TIFF
    functions that return a int value.  This way we do not have to check
    for error status in each wrapping function and an exception will always be
    appropriately raised.
    """
    if status == 0:
        raise RuntimeError('failed')
