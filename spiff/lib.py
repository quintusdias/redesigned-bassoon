# Standard library imports
import ctypes
import datetime
from enum import IntEnum
import queue
import warnings

# Third party library imports
import numpy as np

# Local imports
from . import config, lib
from .tags import TAGS

_LIBTIFF, _LIBC = config.load_libraries('tiff', 'c')


class LibTIFFError(RuntimeError):
    """
    Raise this exception if we detect a generic error from libtiff.
    """
    pass


# The error messages queue
EQ = queue.Queue()


def _set_error_warning_handlers():
    """
    Setup default python error and warning handlers.
    """
    old_warning_handler = setWarningHandler()
    old_error_handler = setErrorHandler()

    return old_error_handler, old_warning_handler


def _reset_error_warning_handlers(old_error_handler, old_warning_handler):
    """
    Restore previous error and warning handlers.
    """
    setWarningHandler(old_warning_handler)
    setErrorHandler(old_error_handler)


def _handle_error(module, fmt, ap):
    # Use VSPRINTF in the C library to put together the error message.
    # int vsprintf(char * buffer, const char * restrict format, va_list ap);
    buffer = ctypes.create_string_buffer(1000)

    argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
    _LIBC.vsprintf.argtypes = argtypes
    _LIBC.vsprintf.restype = ctypes.c_int32
    _LIBC.vsprintf(buffer, fmt, ap)

    module = module.decode('utf-8')
    error_str = buffer.value.decode('utf-8')

    message = f"{module}: {error_str}"
    EQ.put(message)
    return None


def _handle_warning(module, fmt, ap):
    # Use VSPRINTF in the C library to put together the warning message.
    # int vsprintf(char * buffer, const char * restrict format, va_list ap);
    buffer = ctypes.create_string_buffer(1000)

    argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
    _LIBC.vsprintf.argtypes = argtypes
    _LIBC.vsprintf.restype = ctypes.c_int32
    _LIBC.vsprintf(buffer, fmt, ap)

    module = module.decode('utf-8')
    warning_str = buffer.value.decode('utf-8')

    message = f"{module}: {warning_str}"
    warnings.warn(message)


# Set the function types for the warning handler.
_WFUNCTYPE = ctypes.CFUNCTYPE(
    ctypes.c_void_p,  # return type of warning handler, void *
    ctypes.c_char_p,  # module
    ctypes.c_char_p,  # fmt
    ctypes.c_void_p  # va_list
)

_ERROR_HANDLER = _WFUNCTYPE(_handle_error)
_WARNING_HANDLER = _WFUNCTYPE(_handle_warning)


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
    Enumeration corresponding to EXTRASAMPLE_* values listed in tiff.h

    Examples
    --------
    >>> import numpy as np
    >>> import skimage.data
    >>> from spiff import TIFF, lib
    >>> gray = skimage.data.camera().reshape((512, 512, 1))
    >>> # Create a gradient alpha layer.
    >>> x = np.arange(0, 256, 0.5).astype(np.uint8).reshape(512, 1)
    >>> alpha = np.repeat(x, 512, axis=1).reshape((512, 512, 1))
    >>> image = np.concatenate((gray, alpha), axis=2)
    >>> w, h, nz = image.shape
    >>> tw, th = int(w/2), int(h/2)
    >>> t = TIFF('camera-extrasamples.tif', mode='w8')
    >>> t['Photometric'] = lib.Photometric.MINISBLACK
    >>> t['Compression'] = lib.Compression.NONE
    >>> t['ImageWidth'] = w
    >>> t['ImageLength'] = h
    >>> t['BitsPerSample'] = 8
    >>> t['SamplesPerPixel'] = nz
    >>> t['PlanarConfig'] = lib.PlanarConfig.CONTIG
    >>> t['TileLength'] = th
    >>> t['TileWidth'] = tw
    >>> t['ExtraSamples'] = (lib.ExtraSamples.ASSOCALPHA, )
    >>> t[:] = image

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

    Examples
    --------
    >>> import numpy as np
    >>> import skimage.data
    >>> from spiff import TIFF, lib
    >>> image = skimage.data.astronaut()
    >>> t = TIFF('astronaut-jpeg.tif', mode='w8')
    >>> t['Photometric'] = lib.Photometric.YCBCR
    >>> t['Compression'] = lib.Compression.JPEG
    >>> t['JPEGColorMode'] = lib.JPEGColorMode.RGB
    >>> t['PlanarConfig'] = lib.PlanarConfig.CONTIG
    >>> t['JPEGQuality'] = 90
    >>> t['YCbCrSubsampling'] = (1, 1)
    >>> w, h, nz = image.shape
    >>> t['ImageWidth'] = w
    >>> t['ImageLength'] = h
    >>> t['TileWidth'] = int(w/2)
    >>> t['TileLength'] = int(h/2)
    >>> t['BitsPerSample'] = 8
    >>> t['SamplesPerPixel'] = nz
    >>> t['Software'] = lib.getVersion()
    >>> t[:] = image
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
    Enumeration corresponding to PREDICTOR* values listed in tiff.h

    Examples
    --------
    >>> import skimage.data
    >>> from spiff import TIFF, lib
    >>> image = skimage.data.astronaut()
    >>> h, w, nz = image.shape
    >>> t = TIFF('astronaut-predictor.tif', mode='w8')
    >>> t['Photometric'] = lib.Photometric.RGB
    >>> t['Compression'] = lib.Compression.LZW
    >>> t['Predictor'] = lib.Predictor.HORIZONTAL
    >>> t['BitsPerSample'] = 8
    >>> t['ImageWidth'] = w
    >>> t['ImageLength'] = h
    >>> t['SamplesPerPixel'] = nz
    >>> t['PlanarConfig'] = lib.PlanarConfig.CONTIG
    >>> t['TileLength'] = int(h/2)
    >>> t['TileWidth'] = int(w/2)
    >>> t['Software'] = 'SPIFF!'
    >>> t[:] = image
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
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p]
    _LIBTIFF.TIFFClose.argtypes = ARGTYPES
    _LIBTIFF.TIFFClose.restype = None
    _LIBTIFF.TIFFClose(fp)

    _reset_error_warning_handlers(err_handler, warn_handler)


def currentDirOffset(fp):
    """
    Corresponds to TIFFCurrentDirOffset
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p]
    _LIBTIFF.TIFFCurrentDirOffset.argtypes = ARGTYPES
    _LIBTIFF.TIFFCurrentDirOffset.restype = ctypes.c_uint64
    offset = _LIBTIFF.TIFFCurrentDirOffset(fp)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return offset


def open(filename, mode='r'):
    """
    Corresponds to TIFFOpen
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_char_p, ctypes.c_char_p]
    _LIBTIFF.TIFFOpen.argtypes = ARGTYPES
    _LIBTIFF.TIFFOpen.restype = ctypes.c_void_p
    file_argument = ctypes.c_char_p(filename.encode())
    mode_argument = ctypes.c_char_p(mode.encode())
    fp = _LIBTIFF.TIFFOpen(file_argument, mode_argument)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return fp


def getVersion():
    _LIBTIFF.TIFFGetVersion.restype = ctypes.c_char_p
    v = _LIBTIFF.TIFFGetVersion()
    return v.decode('utf-8')


def setField(fp, tag, value):
    """
    Corresponds to TIFFSetField
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p, ctypes.c_int32]

    # Append the proper return type for the tag.
    tag_num = TAGS[tag]['number']
    tag_type = TAGS[tag]['type']
    if tag_num == 320:
        # ColorMap
        # One array passed for each sample.
        args = [ctypes.POINTER(ctypes.c_uint16) for _ in range(value.shape[1])]
        ARGTYPES.extend(args)
    elif tag_num == 330:
        # SubIFDs
        ARGTYPES.extend([ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint64)])
    elif tag_num == 333:
        # InkNames
        ARGTYPES.extend([ctypes.c_uint16, ctypes.c_char_p])
    elif tag_num == 338:
        # ExtraSamples
        ARGTYPES.extend([ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16)])
    elif tag_num == 530:
        ARGTYPES.extend(tag_type)
    else:
        ARGTYPES.append(tag_type)
    _LIBTIFF.TIFFSetField.argtypes = ARGTYPES
    _LIBTIFF.TIFFSetField.restype = check_error

    if tag_num == 284 and value == lib.PlanarConfig.SEPARATE:
        msg = (
            "Writing images with planar configuration SEPARATE is not "
            "supported."
        )
        raise NotImplementedError(msg)

    elif tag_num == 320:
        # ColorMap:  split the colormap into uint16 arrays for each sample.
        columns = [value[:, j].astype(np.uint16) for j in range(value.shape[1])]
        red = columns[0].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        green = columns[1].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        blue = columns[2].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        _LIBTIFF.TIFFSetField(fp, tag_num, red, green, blue)

    elif tag_num == 330:
        # SubIFDs:  the array value should just be zeros.  No need for the
        # user to pass anything but the count.
        n = value
        arr = (ctypes.c_uint64 * n)()
        for j in range(n):
            arr[j] = 0

        _LIBTIFF.TIFFSetField(fp, tag_num, n, arr)

    elif tag_num == 333:
        # InkNames
        # Input is an iterable of strings.  Turn it into a null-terminated and
        # null-separated single string.
        inks = '\0'.join(value) + '\0'
        inks = inks.encode('utf-8')
        n = len(inks)
        _LIBTIFF.TIFFSetField(fp, tag_num, n, ctypes.c_char_p(inks))

    elif tag_num == 338:
        # ExtraSamples
        # We pass a count and an array of values.
        try:
            n = len(value)
        except TypeError:
            # singleton?
            n = 1
            value = [value]

        arr = (ctypes.c_uint16 * n)()
        for j in range(n):
            arr[j] = value[j]

        _LIBTIFF.TIFFSetField(fp, tag_num, n, arr)

    elif tag_num == 530:
        _LIBTIFF.TIFFSetField(fp, tag_num, value[0], value[1])
    elif tag_num == 306 and isinstance(value, datetime.datetime):
        value = value.strftime('%Y:%m:%d %H:%M:%S').encode('utf-8')
        _LIBTIFF.TIFFSetField(fp, tag_num, ctypes.c_char_p(value))
    elif tag_type == ctypes.c_char_p:
        value = value.encode('utf-8')
        _LIBTIFF.TIFFSetField(fp, tag_num, ctypes.c_char_p(value))
    else:
        _LIBTIFF.TIFFSetField(fp, tag_num, value)

    _reset_error_warning_handlers(err_handler, warn_handler)


def computeStrip(fp, row, sample):
    """
    Corresponds to TIFFComputeStrip
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint16]
    _LIBTIFF.TIFFComputeStrip.argtypes = ARGTYPES
    _LIBTIFF.TIFFComputeStrip.restype = ctypes.c_uint32
    stripnum = _LIBTIFF.TIFFComputeStrip(fp, row, sample)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return stripnum


def computeTile(fp, x, y, z, sample):
    """
    Corresponds to TIFFComputeTile
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint16]
    _LIBTIFF.TIFFComputeTile.argtypes = ARGTYPES
    _LIBTIFF.TIFFComputeTile.restype = ctypes.c_uint32
    tilenum = _LIBTIFF.TIFFComputeTile(fp, x, y, z, sample)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return tilenum


def readEncodedStrip(fp, stripnum, strip):
    """
    Corresponds to TIFFReadEncodedStrip
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32
    ]
    _LIBTIFF.TIFFReadEncodedStrip.argtypes = ARGTYPES
    _LIBTIFF.TIFFReadEncodedStrip.restype = check_error
    _LIBTIFF.TIFFReadEncodedStrip(fp, stripnum,
                                  strip.ctypes.data_as(ctypes.c_void_p), -1)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return strip


def readEncodedTile(fp, tilenum, tile):
    """
    Corresponds to TIFFComputeTile
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                ctypes.c_int32]
    _LIBTIFF.TIFFReadEncodedTile.argtypes = ARGTYPES
    _LIBTIFF.TIFFReadEncodedTile.restype = check_error
    _LIBTIFF.TIFFReadEncodedTile(fp, tilenum,
                                 tile.ctypes.data_as(ctypes.c_void_p), -1)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return tile


def getField(fp, tag):
    """
    Corresponds to TIFFGetField in the TIFF library.
    """
    import pdb; pdb.set_trace()
    err_handler, warn_handler = _set_error_warning_handlers()

    tag_num = TAGS[tag]['number']

    if tag_num == 320:
        # ColorMap
        import pdb; pdb.set_trace()
        value = _getField_colormap(fp, tag)
    else:
        value = _getField_default(fp, tag)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return value


def _getField_colormap(fp, tag):
    """
    """
    import pdb; pdb.set_trace()
    _argtypes = [ctypes.POINTER(ctypes.c_uint16),
                 ctypes.POINTER(ctypes.c_uint16),
                 ctypes.POINTER(ctypes.c_uint16)]
    _LIBTIFF.TIFFGetField.argtypes = ARGTYPES

    _LIBTIFF.TIFFGetField.restype = check_error

    bps = lib.getFieldDefaulted(fp, 'BitsPerSample')
    n = 1 << bps
    red = np.zeros((n, 1), dtype=np.uint16)
    green = np.zeros((n, 1), dtype=np.uint16)
    blue = np.zeros((n, 1), dtype=np.uint16)
    _LIBTIFF.TIFFGetField(fp, tag_num,
                              red.ctypes.data_as(np.uint16),
                              green.ctypes.data_as(np.uint16),
                              blue.ctypes.data_as(np.uint16))
    colormap = np.zeros((n, 3), dtype=np.uint16)
    colormap[:, 0] = red
    colormap[:, 1] = green
    colormap[:, 2] = blue
    return colormap

def _getField_default(fp, tag):
    """
    Corresponds to TIFFGetField in the TIFF library.
    """
    ARGTYPES = [ctypes.c_void_p, ctypes.c_int32]

    tag_num = TAGS[tag]['number']

    # Append the proper return type for the tag.
    ARGTYPES.append(_argtypes)
    _LIBTIFF.TIFFGetField.argtypes = ARGTYPES

    _LIBTIFF.TIFFGetField.restype = check_error

    # instantiate the tag value
    item = tag_type()
    _LIBTIFF.TIFFGetField(fp, tag_num, ctypes.byref(item))

    _reset_error_warning_handlers(err_handler, warn_handler)

    return item.value


def getFieldDefaulted(fp, tag):
    """
    Corresponds to the TIFFGetFieldDefaulted library routine.
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p, ctypes.c_int32]

    tag_num = TAGS[tag]['number']

    # Append the proper return type for the tag.
    tag_type = TAGS[tag]['type']
    ARGTYPES.append(ctypes.POINTER(TAGS[tag]['type']))
    _LIBTIFF.TIFFGetFieldDefaulted.argtypes = ARGTYPES

    _LIBTIFF.TIFFGetFieldDefaulted.restype = check_error

    # instantiate the tag value
    item = tag_type()
    _LIBTIFF.TIFFGetFieldDefaulted(fp, tag_num, ctypes.byref(item))

    _reset_error_warning_handlers(err_handler, warn_handler)

    return item.value


def isTiled(fp):
    """
    Corresponds to TIFFIsTiled
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p]

    _LIBTIFF.TIFFIsTiled.argtypes = ARGTYPES
    _LIBTIFF.TIFFIsTiled.restype = ctypes.c_int

    status = _LIBTIFF.TIFFIsTiled(fp)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return status


def numberOfDirectories(fp):
    """
    Corresponds to TIFFNumberOfDirectories.
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p]
    _LIBTIFF.TIFFNumberOfDirectories.argtypes = ARGTYPES
    _LIBTIFF.TIFFNumberOfDirectories.restype = ctypes.c_uint16

    n = _LIBTIFF.TIFFNumberOfDirectories(fp)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return n


def numberOfStrips(fp):
    """
    Corresponds to TIFFNumberOfStrips.
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p]
    _LIBTIFF.TIFFNumberOfStrips.argtypes = ARGTYPES
    _LIBTIFF.TIFFNumberOfStrips.restype = ctypes.c_uint32

    n = _LIBTIFF.TIFFNumberOfStrips(fp)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return n


def numberOfTiles(fp):
    """
    Corresponds to TIFFNumberOfTiles.
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p]
    _LIBTIFF.TIFFNumberOfTiles.argtypes = ARGTYPES
    _LIBTIFF.TIFFNumberOfTiles.restype = ctypes.c_uint32

    numtiles = _LIBTIFF.TIFFNumberOfTiles(fp)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return numtiles


def readEXIFDirectory(fp, offset):
    """
    Corresponds to TIFFReadEXIFIFDDirectory.

    Use this routine only with the ExifIFD tag.
    """
    err_handler, warn_handler = _set_error_warning_handlers()
    ARGTYPES = [ctypes.c_void_p, ctypes.c_uint64]
    _LIBTIFF.TIFFReadEXIFDirectory.argtypes = ARGTYPES
    _LIBTIFF.TIFFReadEXIFDirectory.restype = check_error
    _LIBTIFF.TIFFReadEXIFDirectory(fp, offset)
    _reset_error_warning_handlers(err_handler, warn_handler)


def readDirectory(fp):
    """
    Corresponds to TIFFReadDirectory.
    """
    err_handler, warn_handler = _set_error_warning_handlers()
    ARGTYPES = [ctypes.c_void_p]
    _LIBTIFF.TIFFReadDirectory.argtypes = ARGTYPES
    _LIBTIFF.TIFFReadDirectory.restype = check_error
    _LIBTIFF.TIFFReadDirectory(fp)
    _reset_error_warning_handlers(err_handler, warn_handler)


def setSubDirectory(fp, offset):
    """
    Corresponds to TIFFSetSubDirectory.

    Use this routine with the SubIFDs tag.
    """
    err_handler, warn_handler = _set_error_warning_handlers()
    ARGTYPES = [ctypes.c_void_p, ctypes.c_uint64]
    _LIBTIFF.TIFFSetSubDirectory.argtypes = ARGTYPES
    _LIBTIFF.TIFFSetSubDirectory.restype = check_error
    _LIBTIFF.TIFFSetSubDirectory(fp, offset)
    _reset_error_warning_handlers(err_handler, warn_handler)


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
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int32, ctypes.c_int32
    ]

    _LIBTIFF.TIFFReadRGBAImageOriented.argtypes = ARGTYPES
    _LIBTIFF.TIFFReadRGBAImageOriented.restype = check_error

    img = np.zeros((height, width, 4), dtype=np.uint8)
    raster = img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    _LIBTIFF.TIFFReadRGBAImageOriented(fp, width, height, raster, orientation,
                                       stopOnError)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return img


def RGBAImageOK(fp):
    """
    Corresponds to TIFFRGBAImageOK.
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    emsg = ctypes.create_string_buffer(1024)
    ARGTYPES = [ctypes.c_void_p, ctypes.c_char_p]
    _LIBTIFF.TIFFRGBAImageOK.argtypes = ARGTYPES
    _LIBTIFF.TIFFRGBAImageOK.restype = ctypes.c_int
    ok = _LIBTIFF.TIFFRGBAImageOK(fp, emsg)
    if not ok:
        error_message = f"libtiff:  {emsg.value.decode('utf-8')}"
        raise NotRGBACompatibleError(error_message)

    _reset_error_warning_handlers(err_handler, warn_handler)


def setErrorHandler(func=_ERROR_HANDLER):
    # The signature of the error handler is
    #     const char *module, const char *fmt, va_list ap
    #
    # The return type is void *
    _LIBTIFF.TIFFSetErrorHandler.argtypes = [_WFUNCTYPE]
    _LIBTIFF.TIFFSetErrorHandler.restype = _WFUNCTYPE
    old_error_handler = _LIBTIFF.TIFFSetErrorHandler(func)
    return old_error_handler


def setWarningHandler(func=_WARNING_HANDLER):
    # The signature of the warning handler is
    #     const char *module, const char *fmt, va_list ap
    #
    # The return type is void *
    _LIBTIFF.TIFFSetWarningHandler.argtypes = [_WFUNCTYPE]
    _LIBTIFF.TIFFSetWarningHandler.restype = _WFUNCTYPE
    old_warning_handler = _LIBTIFF.TIFFSetWarningHandler(func)
    return old_warning_handler


def writeDirectory(fp):
    """
    Corresponds to TIFFWriteDirectory.
    """
    err_handler, warn_handler = _set_error_warning_handlers()
    ARGTYPES = [ctypes.c_void_p]
    _LIBTIFF.TIFFWriteDirectory.argtypes = ARGTYPES
    _LIBTIFF.TIFFWriteDirectory.restype = check_error
    _LIBTIFF.TIFFWriteDirectory(fp)
    _reset_error_warning_handlers(err_handler, warn_handler)


def writeEncodedStrip(fp, stripnum, stripdata, size=-1):
    """
    Corresponds to TIFFWriteEncodedStrip.
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32
    ]
    _LIBTIFF.TIFFWriteEncodedStrip.argtypes = ARGTYPES
    _LIBTIFF.TIFFWriteEncodedStrip.restype = check_error
    raster = stripdata.ctypes.data_as(ctypes.c_void_p)
    _LIBTIFF.TIFFWriteEncodedStrip(fp, stripnum, raster, size)

    _reset_error_warning_handlers(err_handler, warn_handler)


def writeEncodedTile(fp, tilenum, tiledata, size=-1):
    """
    Corresponds to TIFFWriteEncodedTile.
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32
    ]
    _LIBTIFF.TIFFWriteEncodedTile.argtypes = ARGTYPES
    _LIBTIFF.TIFFWriteEncodedTile.restype = check_error
    raster = tiledata.ctypes.data_as(ctypes.c_void_p)
    _LIBTIFF.TIFFWriteEncodedTile(fp, tilenum, raster, size)

    _reset_error_warning_handlers(err_handler, warn_handler)


def check_error(status):
    """
    Set a generic function as the restype attribute of all TIFF
    functions that return a int value.  This way we do not have to check
    for error status in each wrapping function and an exception will always be
    appropriately raised.
    """
    msg = ''
    while not EQ.empty():
        msg = EQ.get()
        raise LibTIFFError(msg)

    if status == 0:
        raise RuntimeError('failed')
