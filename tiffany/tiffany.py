import struct

# 3rd party libraries
import numpy as np

from . import lib
from . import tags


class TIFF(object):
    """
    Attributes
    ----------
    datatype2fmt : dict
        Map the TIFF entry datatype to something that can be used by the struct
        module.
    tagnum2name : dict
        Map the tag number to a tag name.
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
                    12: ('d', 8)}

    def __init__(self, filename, mode='r'):
        self.tfp = lib.open(filename, mode=mode)

        self.tags = {}

        if 'w' in mode:
            self.fp = None
        else:
            self.fp = open(filename, mode='rb')
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
                lib.writeEncodedStrip(self.tfp, stripnum, strip)

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
                lib.writeEncodedTile(self.tfp, tilenum, tile)

    def __setitem__(self, idx, value):
        """
        Set a tag value or write part/all of an image.
        """
        if idx in self.tagnum2name.values():

            # Setting a TIFF tag...
            lib.setField(self.tfp, idx, value)
            self.tags[idx] = value

        elif isinstance(idx, slice):
            if idx.start is None and idx.step is None and idx.stop is None:
                # Case of t[:] = ...
                if lib.isTiled(self.tfp):
                    self._writeTiledImage(value)
                else:
                    self._writeStrippedImage(value)
        else:
            msg = f"Unhandled:  {idx}"
            raise RuntimeError(msg)

    def __getitem__(self, idx):
        """
        Either retrieve a named tag or read part/all of an image.
        """
        if isinstance(idx, slice):
            if lib.isTiled(self.tfp):
                shape = (self['imagelength'], self['imagewidth'])
                image = np.zeros(shape, dtype=np.uint8)
                height, width = self['imagelength'], self['imagelength']
                theight, twidth = self['tilelength'], self['tilelength']
                for row in range(0, height, theight):
                    rslice = slice(row, row + theight)
                    for col in range(0, width, twidth):
                        tilenum = lib.computeTile(self.tfp, col, row, 0)
                        cslice = slice(col, col + twidth)
                        tile = lib.readEncodedTile(self.tfp, tilenum)
                        image[rslice, cslice] = tile
                return image
            else:
                msg = f"Strips with t[:] = ... is not handled"
                raise RuntimeError(msg)

            if idx.start is None and idx.stop is None and idx.step is None:
                # case is [:]
                img = lib.readRGBAImage(self.tfp,
                                        width=self.tags['imagewidth'],
                                        height=self.tags['imagelength'])
                return img
        elif isinstance(idx, str):
            return self.tags[idx]

    def parse_ifd(self):
        """
        Parse the TIFF for metadata.

        Parameters
        ----------
        fp : file-like object
            The TIFF file to read.
        """
        buffer = self.fp.read(2)
        num_tags, = struct.unpack(f"{self.endian}H", buffer)

        buffer = self.fp.read(num_tags * 12)

        tags = {}
        for idx in range(num_tags):
            tag, value = self._process_entry(buffer[idx * 12:(idx + 1) * 12])
            tags[tag] = value
        self.tags = tags

        buffer = self.fp.read(4)
        self.next_offset, = struct.unpack(f"{self.endian}I", buffer)

    def _process_entry(self, buffer):
        """
        Read an IFD entry from the buffer.

        Returns
        -------
        tag, value
        """
        fmt = self.endian + 'HHII'
        tag_num, dtype, count, offset = struct.unpack(fmt, buffer)

        try:
            fmt = self.datatype2fmt[dtype][0] * count
            payload_size = self.datatype2fmt[dtype][1] * count
        except KeyError:
            msg = 'Invalid TIFF tag datatype ({dtype}).'
            raise IOError(msg)

        if payload_size <= 4:
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
        buffer = self.fp.read(8)

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

        _, offset = struct.unpack(self.endian + 'HI', buffer[2:8])
        self.fp.seek(offset)
