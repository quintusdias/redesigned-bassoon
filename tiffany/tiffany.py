import struct

# 3rd party libraries
import numpy as np

from . import lib
from . import tags


class JPEGColorModeRawError(RuntimeError):
    """
    Raise this exception if an attempt is made to write YCbCr/JPEG images with
    JPEGCOLORMODERAW instead of JPEGCOLORMODERGB.
    """
    pass


class TIFF(object):
    """
    Attributes
    ----------
    datatype2fmt : dict
        Map the TIFF entry datatype to something that can be used by the struct
        module.
    tagnum2name : dict
        Map the tag number to a tag name.
    bigtiff: bool
        True if BigTIFF, false for Classic TIFF
    """
    tagnum2name = tags.tagnum2name

    datatype2fmt = {1: ('B', 1),
                    2: ('B', 1),
                    3: ('H', 2),
                    4: ('I', 4),
                    5: ('II', 8),
                    7: ('B', 1),
                    9: ('i', 4),
                    10: ('ii', 8),
                    11: ('f', 4),
                    12: ('d', 8),
                    16: ('Q', 8)}

    def __init__(self, path, mode='r'):
        self.path = path
        self.tfp = lib.open(str(path), mode=mode)

        self.tags = {}

        if 'w' in mode:
            self.fp = None
        else:
            self.fp = self.path.open(mode='rb')
            self.parse_header()
            self.parse_ifd()

    def __del__(self):
        """
        Perform any needed resource clean-up.
        """
        # Close the Python file pointer.
        if self.fp is not None:
            self.fp.close()

        # Close the TIFF file pointer.
        lib.close(self.tfp)

    def _writeStrippedImage(self, image):
        """
        Write an entire stripped image.
        """
        numstrips = int(self['imagelength'] / self['rowsperstrip'])
        stripnum = -1
        for r in range(numstrips):
            rslice = slice(r * self['rowsperstrip'],
                           (r + 1) * self['rowsperstrip'])
            strip = image[rslice, :].copy()
            stripnum += 1
            lib.writeEncodedStrip(self.tfp, stripnum, strip, size=strip.nbytes)

    def _writeTiledImage(self, image):
        """
        Write an entire tiled image.
        """
        # numtiles = lib.numberOfTiles(self.tfp)
        numtilerows = int(self['imagelength'] / self['tilelength'])
        numtilecols = int(self['imagewidth'] / self['tilewidth'])
        tilenum = -1
        for r in range(numtilerows):
            rslice = slice(r * self['tilelength'],
                           (r + 1) * self['tilelength'])
            for c in range(numtilecols):
                cslice = slice(c * self['tilewidth'],
                               (c + 1) * self['tilewidth'])
                tile = image[rslice, cslice].copy()
                tilenum += 1
                lib.writeEncodedTile(self.tfp, tilenum, tile, size=tile.nbytes)

    def __setitem__(self, idx, value):
        """
        Set a tag value or write part/all of an image.
        """
        if idx in self.tagnum2name.values():

            # Setting a TIFF tag...
            lib.setField(self.tfp, idx, value)
            self.tags[idx] = value

        elif isinstance(idx, slice):
            if (((self['photometric'] == lib.Photometric.YCBCR) and
                 (self['compression'] == lib.Compression.JPEG) and
                 (self['jpegcolormode'] == lib.JPEGColorMode.RAW))):
                msg = (
                    "You must set the jpegcolormode tag to "
                    "JPEGColorMode.RGB in order to write to a YCbCr/JPEG "
                    "image."
                )
                raise JPEGColorModeRawError(msg)

            if idx.start is None and idx.step is None and idx.stop is None:
                # Case of t[:] = ...
                if lib.isTiled(self.tfp):
                    self._writeTiledImage(value)
                else:
                    self._writeStrippedImage(value)
        else:
            msg = f"Unhandled:  {idx}"
            raise RuntimeError(msg)

    def _readStrippedImage(self, idx):
        """
        Read entire image where the orientation is stripped.
        """
        shape = (
            self['imagelength'], self['imagewidth'], self['samplesperpixel']
        )
        image = np.zeros(shape, dtype=np.uint8)
        height = self['imagelength']
        stripheight = self['rowsperstrip']
        for row in range(0, height, stripheight):
            rslice = slice(row, row + stripheight)
            stripnum = lib.computeStrip(self.tfp, row, 0)
            strip = lib.readEncodedStrip(self.tfp, stripnum)
            image[rslice, :, :] = strip

        if self['samplesperpixel'] == 1:
            # squash the trailing dimension of 1.
            image = np.squeeze(image)

        return image

    def _readTiledImage(self, idx):
        """
        Helper routine for assembling an entire image out of tiles.
        """
        shape = (
            self['imagelength'], self['imagewidth'], self['samplesperpixel']
        )
        image = np.zeros(shape, dtype=np.uint8)
        height, width = self['imagelength'], self['imagewidth']
        theight, twidth = self['tilelength'], self['tilewidth']
        for row in range(0, height, theight):
            rslice = slice(row, row + theight)
            for col in range(0, width, twidth):
                tilenum = lib.computeTile(self.tfp, col, row, 0)
                cslice = slice(col, col + twidth)
                tile = lib.readEncodedTile(self.tfp, tilenum)
                image[rslice, cslice, :] = tile

        if self['samplesperpixel'] == 1:
            # squash the trailing dimension of 1.
            image = np.squeeze(image)

        return image

    def __getitem__(self, idx):
        """
        Either retrieve a named tag or read part/all of an image.
        """
        if isinstance(idx, slice):
            if lib.isTiled(self.tfp):
                return self._readTiledImage(slice)
            else:
                return self._readStrippedImage(slice)

            if idx.start is None and idx.stop is None and idx.step is None:
                # case is [:]
                img = lib.readRGBAImage(self.tfp,
                                        width=self.tags['imagewidth'],
                                        height=self.tags['imagelength'])
                return img
        elif isinstance(idx, str):
            if idx == 'jpegcolormode':
                # This is a pseudo-tag that the user might not have set.
                return lib.getFieldDefaulted(self.tfp, 'jpegcolormode')
            return self.tags[idx]

    def parse_ifd(self):
        """
        Parse the TIFF for metadata.

        Parameters
        ----------
        fp : file-like object
            The TIFF file to read.
        """

        if self.bigtiff:
            buffer = self.fp.read(8)
            num_tags, = struct.unpack(f"{self.endian}Q", buffer)
            nb = 20
            buffer = self.fp.read(num_tags * 20 + 8)
        else:
            buffer = self.fp.read(2)
            num_tags, = struct.unpack(f"{self.endian}H", buffer)
            nb = 12
            buffer = self.fp.read(num_tags * 12 + 4)

        tags = {}
        for idx in range(num_tags):
            tag, value = self._process_entry(buffer[idx * nb:(idx + 1) * nb])
            tags[tag] = value
        self.tags = tags

        # Last 4 or 8 bytes contain the offset to the next IFD.
        self.next_offset, = struct.unpack(f"{self.endian}{self.offset_format}",
                                          buffer[nb * num_tags:])

    def _process_entry(self, buffer):
        """
        Read an IFD entry from the buffer.

        Returns
        -------
        tag, value
        """
        fmt = f"{self.endian}{self.tag_entry_format}"
        tag_num, dtype, count, offset = struct.unpack(fmt, buffer)

        try:
            fmt = self.datatype2fmt[dtype][0] * count
            payload_size = self.datatype2fmt[dtype][1] * count
        except KeyError:
            msg = f'Invalid TIFF tag datatype ({dtype}).'
            raise IOError(msg)

        if self.bigtiff and payload_size <= 8:
            # Interpret the payload from the 8 bytes in the tag entry.
            target_buffer = buffer[12:12 + payload_size]
        elif not self.bigtiff and payload_size <= 4:
            # Interpret the payload from the 4 bytes in the tag entry.
            target_buffer = buffer[8:8 + payload_size]
        else:
            # Interpret the payload at the offset specified by the 4 bytes in
            # the tag entry.
            orig = self.fp.tell()
            self.fp.seek(offset)
            target_buffer = self.fp.read(payload_size)
            self.fp.seek(orig)

        if dtype == 2:
            # ASCII
            payload = target_buffer.decode('utf-8').rstrip('\x00')

        else:
            payload = struct.unpack(self.endian + fmt, target_buffer)
            if dtype == 5 or dtype == 10:
                # Rational or Signed Rational.  Construct the list of values.
                rational_payload = []
                for j in range(count):
                    value = float(payload[j * 2]) / float(payload[j * 2 + 1])
                    rational_payload.append(value)
                payload = rational_payload
            if count == 1:
                # If just a single value, then return a scalar instead of a
                # tuple.
                payload = payload[0]

        tag_name = self.tagnum2name[tag_num]
        return tag_name, payload

    def parse_header(self):
        """
        Parse the TIFF header.

        Parameters
        ----------
        fp : file-like object
            The TIFF file to read.
        """
        buffer = self.fp.read(16)

        # First 8 should be (73, 73, 42, 8) or (77, 77, 42, 8)
        data = struct.unpack('BB', buffer[0:2])
        if data[0] == 73 and data[1] == 73:
            # little endian
            self.endian = '<'
        elif data[0] == 77 and data[1] == 77:
            # big endian
            self.endian = '>'
        else:
            msg = (
                f"The byte order indication in the TIFF header "
                f"({buffer[0:2]}) is invalid.  It should be either "
                f"{bytes([73, 73])} or {bytes([77, 77])}."
            )
            raise IOError(msg)

        tifftype, = struct.unpack(f"{self.endian}H", buffer[2:4])
        if tifftype == 42:
            self.bigtiff = False
            self.num_tags_format = 'H'
            self.offset_format = 'I'
            self.tag_entry_format = 'HHII'
        elif tifftype == 43:
            self.bigtiff = True
            self.num_tags_format = 'Q'
            self.offset_format = 'Q'
            self.tag_entry_format = 'HHQQ'
        else:
            msg = (
                f"The file is not recognized as either classic TIFF or "
                f"BigTIFF. "
            )
            raise IOError(msg)

        if self.bigtiff:
            _, _, offset = struct.unpack(f"{self.endian}HHQ", buffer[4:16])
        else:
            offset, = struct.unpack(f"{self.endian}I", buffer[4:8])

        self.fp.seek(offset)
