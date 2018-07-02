# Standard libaries
import datetime as dt
import io
import pathlib
import pprint
import struct

# 3rd party libraries
from lxml import etree
import numpy as np

# Local imports
from . import lib
from . import tags
from . import _cytiff


class DatatypeMismatchError(RuntimeError):
    """
    Raise this exception if an attempt is made to write YCbCr/JPEG images with
    image data that is not uint8.
    """
    pass


class JPEGColorModeRawError(RuntimeError):
    """
    Raise this exception if an attempt is made to write YCbCr/JPEG images with
    JPEGCOLORMODERAW instead of JPEGCOLORMODERGB.
    """
    pass


class NoEXIFIFDError(RuntimeError):
    """
    Raise this exception if the user tries to change to an EXIF IFD and there
    is no EXIF IFD.
    """
    pass


class TIFFReadImageError(RuntimeError):
    """
    Raise this exception if a read operation was inappropriate.  Maybe the
    library will segfault otherwise?
    """
    pass


class TIFF(object):
    """
    Attributes
    ----------
    datatype2fmt : dict
        Map the TIFF entry datatype to something that can be used by the struct
        module.
    rgba : bool
        If true, use the RGBA interface to read the image.
    tagnum2name : dict
        Map the tag number to a tag name.
    bigtiff: bool
        True if BigTIFF, false for Classic TIFF
    """
    tagnum2name = tags.tagnum2name

    # Map the enumerated TIFF datatypes to python.
    datatype2fmt = {
        1: ('B', 1),
        2: ('B', 1),
        3: ('H', 2),
        4: ('I', 4),
        5: ('II', 8),
        7: ('B', 1),
        9: ('i', 4),
        10: ('ii', 8),
        11: ('f', 4),
        12: ('d', 8),
        13: ('I', 4),
        16: ('Q', 8),
        18: ('Q', 8)
    }

    def __init__(self, path, mode='r'):
        """
        Parameters
        ----------
        path : path or string
            Path to TIFF file
        mode : str
            File access mode.
        """

        if isinstance(path, str):
            self.path = pathlib.Path(path)
        else:
            self.path = path
        self.tfp = lib.open(str(path), mode=mode)

        self.tags = {}

        if 'w' in mode:
            self.fp = None
        else:
            self._rgba = False
            self.fp = self.path.open(mode='rb')
            self.parse_header()
            self.parse_ifd()

        self._ifd_offsets = []

    def __iter__(self):
        """
        We are our own iterator.
        """
        self._first_iteration = True
        return self

    def __next__(self):
        """
        Return the next IFD.  That really just means to position this object
        at the next IFD.
        """
        if self._first_iteration:
            # The first time we iterate, just return ourself in our current
            # state.  Otherwise the iteration misses the first image.  Of
            # course, we have to flag so that we don't keep doing that.
            self._first_iteration = False
            return self

        if self.next_offset == 0:
            # We are done, cannot go on to the next image because there isn't
            # one.
            raise StopIteration()

        old_offset = lib.currentDirOffset(self.tfp)
        self._ifd_offsets.append(old_offset)

        # Go to the next IFD, both in libtiff-land, and in python-file-world.
        lib.readDirectory(self.tfp)
        self.fp.seek(self.next_offset)
        self.parse_ifd()
        return self

    def __len__(self):
        return lib.numberOfDirectories(self.tfp)

    def __str__(self):
        s = io.StringIO()
        pp = pprint.PrettyPrinter(stream=s, indent=4)
        pp.pprint(self.tags)
        return s.getvalue()

    def __repr__(self):
        """
        Use the TIFFPrintDirectory library routine for this.

        Sometimes it tacks on unprintable characters, so we have to
        specifically ignore them.
        """
        b = _cytiff.print_directory(self.tfp)
        s = b.decode('utf-8', 'ignore')
        return s

    def __del__(self):
        """
        Perform any needed resource clean-up.
        """
        # Close the Python file pointer.
        if self.fp is not None:
            self.fp.close()

        # Close the TIFF file pointer.
        lib.close(self.tfp)

    @property
    def rgba(self):
        return self._rgba

    @rgba.setter
    def rgba(self, value):
        """
        Parameters
        ----------
        value
            Set to True if we wish to use the RGBA interface to read the image.
        """
        # First check if this is even ok to try.
        lib.RGBAImageOK(self.tfp)

        # Ok, we can proceed.
        self._rgba = value

    @property
    def shape(self):
        if self.spp == 1:
            # It is a 2D image.
            try:
                return (self.h, self.w)
            except KeyError:
                # We can assume that ImageLength and ImageWidth are not yet
                # defined.  shape is meaningless at this point.
                msg = "Either ImageLength or ImageWidth is not yet defined."
                raise RuntimeError(msg)
        else:
            # 3D image.
            return (self.h, self.w, self.spp)

    @property
    def h(self):
        """
        Shortcut for the image height.
        """
        return self['ImageLength']

    @property
    def th(self):
        """
        Shortcut for the image tile height.
        """
        return self['TileLength']

    @property
    def w(self):
        """
        Shortcut for the image width.
        """
        return self['ImageWidth']

    @property
    def tw(self):
        """
        Shortcut for the image tile width.
        """
        return self['TileWidth']

    @property
    def bps(self):
        """
        Shortcut for the image bits per sample.

        TIFFs will store more than one value if there is more than one image
        plane.  We assume that only one value is needed.
        """
        try:
            return self['BitsPerSample'][0]
        except TypeError:
            return self['BitsPerSample']

    @property
    def rps(self):
        """
        Shortcut for the image rows per strip
        """
        return self['RowsPerStrip']

    @property
    def sf(self):
        """
        Shortcut for the sample format
        """
        return self['SampleFormat']

    @property
    def spp(self):
        """
        Shortcut for the image depth
        """
        try:
            return self['SamplesPerPixel']
        except KeyError:
            # It's possible that SamplesPerPixel isn't defined yet, or maybe
            # is not even written to the file.  Try to get the defaulted value.
            return lib.getFieldDefaulted(self.tfp, 'SamplesPerPixel')

    def new_image(self):
        """
        Initialize the next image in this multi-page tiff.
        """
        old_offset = lib.currentDirOffset(self.tfp)
        self._ifd_offsets.append(old_offset)
        lib.writeDirectory(self.tfp)

    def back(self):
        """
        Go back to the previous IFD.
        """
        old_offset = self._ifd_offsets.pop()
        lib.setSubDirectory(self.tfp, old_offset)

        # And finally, refresh the tags.
        self.fp.seek(old_offset)
        self.parse_ifd()

    def visit_ifd(self, offset):
        """
        Change directories and read the contents of a new IFD.

        Parameters
        ----------
        offset : unsigned integer
            Offset to the sub ifd.  This should be a value retrieved from the
            ExifIFD, GPSIfd, or SubIFDs tags.  Providing an incorrect value is
            a good way to segfault libtiff.
        """
        old_offset = lib.currentDirOffset(self.tfp)
        if 'ExifIFD' in self.tags.keys() and self.tags['ExifIFD'] == offset:
            lib.readEXIFDirectory(self.tfp, offset)
        else:
            lib.setSubDirectory(self.tfp, offset)

        # After we've successfully transfered to the new IFD, save the old
        # offset.
        self._ifd_offsets.append(old_offset)

        # And finally, refresh the tags.
        self.fp.seek(offset)
        self.parse_ifd()

    def _writeStrippedImage(self, image):
        """
        Write an entire stripped image.
        """
        numstrips = lib.numberOfStrips(self.tfp)
        for row in range(0, self.h, self.rps):
            stripnum = lib.computeStrip(self.tfp, row, 0)
            rslice = slice(row, row + self.rps)

            strip = image[rslice, :].copy()

            # Is it the last strip?  Is that last strip a full strip?
            # If not, then we need to pad it.
            if stripnum == (numstrips - 1) and self.h % self.rps > 0:
                if self.spp > 1:
                    shape = rslice.stop - self.h, self.w, self.spp
                else:
                    shape = rslice.stop - self.h, self.w
                arrs = strip, np.zeros(shape, dtype=strip.dtype)
                strip = np.concatenate(arrs, axis=0)

            lib.writeEncodedStrip(self.tfp, stripnum, strip, size=strip.nbytes)

    def _writeTiledImage(self, image):
        """
        Write an entire tiled image.
        """
        numtilerows = int(round(self.h / self.th + 0.5))
        numtilecols = int(round(self.w / self.tw + 0.5))

        tilerow = -1

        for row in range(0, self.h, self.th):

            tilerow += 1

            rslice = slice(row, row + self.th)

            tilecol = -1

            for col in range(0, self.w, self.tw):

                tilecol += 1

                cslice = slice(col, col + self.tw)
                tilenum = lib.computeTile(self.tfp, col, row, 0, 0)
                tile = image[rslice, cslice].copy()

                if self.w % self.tw > 0 and tilecol == (numtilecols - 1):
                    # If the tile dimensions don't evenly partition the image
                    # and if the tile column is at the end, then the tile
                    # right now is truncated.  Extend it on the right hand
                    # side to a full tile.
                    if self.spp > 1:
                        shape = (tile.shape[0], cslice.stop - self.w, self.spp)
                    else:
                        shape = (tile.shape[0], cslice.stop - self.w)
                    padright = np.zeros(shape, dtype=image.dtype)
                    tile = np.concatenate((tile, padright), axis=1)

                if self.h % self.th > 0 and tilerow == (numtilerows - 1):
                    # If the tile dimensions don't evenly partition the image
                    # and if the tile row is at the end, the image tile is
                    # truncated.  Extend it on the bottom side to a full
                    # tile.
                    if self.spp > 1:
                        shape = (rslice.stop - self.h, tile.shape[1], self.spp)
                    else:
                        shape = (rslice.stop - self.h, tile.shape[1])
                    padbottom = np.zeros(shape, dtype=image.dtype)
                    tile = np.concatenate((tile, padbottom), axis=0)

                lib.writeEncodedTile(self.tfp, tilenum, tile, size=tile.nbytes)

    def _write_image(self, idx, image):
        if (((self['Photometric'] == lib.Photometric.YCBCR) and
             (self['Compression'] == lib.Compression.JPEG))):

            # JPEG has some restrictions.
            if self['JPEGColorMode'] == lib.JPEGColorMode.RAW:
                msg = (
                    "You must set the JPEGColorMode tag to "
                    "JPEGColorMode.RGB in order to write to a YCbCr/JPEG "
                    "image."
                )
                raise JPEGColorModeRawError(msg)

            if image.dtype != np.uint8:
                msg = (
                    f"Writing JPEG images with datatype {image.dtype} are not "
                    f"supported.  They must be uint8."
                )
                raise DatatypeMismatchError(msg)

        if idx.start is None and idx.step is None and idx.stop is None:
            # Case of t[:] = ...
            if lib.isTiled(self.tfp):
                self._writeTiledImage(image)
            else:
                self._writeStrippedImage(image)

    def __setitem__(self, idx, value):
        """
        Set a tag value or write part/all of an image.

        InkNames : Should be an iterable of strings.
        """
        if idx in self.tagnum2name.values():

            # Setting a TIFF tag...
            lib.setField(self.tfp, idx, value)
            self.tags[idx] = value

        elif isinstance(idx, slice):
            self._write_image(idx, value)
        else:
            msg = f"Unhandled:  {idx}"
            raise RuntimeError(msg)

    def _determine_datatype(self):
        """
        Determine the datatype for incoming imagery.
        """
        if self.bps == 8 and self.sf == lib.SampleFormat.UINT:
            return np.uint8
        elif self.bps == 8 and self.sf == lib.SampleFormat.INT:
            return np.int8
        elif self.bps == 16 and self.sf == lib.SampleFormat.UINT:
            return np.uint16
        elif self.bps == 16 and self.sf == lib.SampleFormat.INT:
            return np.int16
        elif self.bps == 32 and self.sf == lib.SampleFormat.UINT:
            return np.uint32
        elif self.bps == 32 and self.sf == lib.SampleFormat.INT:
            return np.int32
        elif self.bps == 32 and self.sf == lib.SampleFormat.IEEEFP:
            return np.float32
        elif self.bps == 64 and self.sf == lib.SampleFormat.UINT:
            return np.uint64
        elif self.bps == 64 and self.sf == lib.SampleFormat.INT:
            return np.int64
        elif self.bps == 64 and self.sf == lib.SampleFormat.IEEEFP:
            return np.float64

    def _readStrippedImage(self, idx):
        """
        Read entire image where the orientation is stripped.
        """
        if self['PlanarConfig'] == lib.PlanarConfig.CONTIG:
            return self._readStrippedContigImage(idx)
        else:
            return self._readStrippedSeparateImage(idx)

    def _readStrippedSeparateImage(self, idx):
        """
        Read entire image where the orientation is stripped and the planar
        configuration is separate.
        """
        numstrips = lib.numberOfStrips(self.tfp)
        strips_per_plane = numstrips // self.spp

        shape = self.h, self.w, self.spp
        dtype = self._determine_datatype()
        image = np.zeros(shape, dtype=dtype)

        stripshape = (self.rps, self.w)

        for row in range(0, self.h, self.rps):
            rslice = slice(row, row + self.rps)

            for plane in range(0, self.spp):

                stripnum = lib.computeStrip(self.tfp, row, plane)
                strip = np.zeros(stripshape, dtype=dtype)
                lib.readEncodedStrip(self.tfp, stripnum, strip)

                # Are these strips on the bottom?  Are they full strips?
                # If not, then we need to shave off some rows.
                if (stripnum + 1) % strips_per_plane == 0:
                    if self.h % self.rps > 0:
                        strip = strip[:self.h % self.rps, :]

                image[rslice, :, plane] = strip

        if self['SamplesPerPixel'] == 1:
            # squash the trailing dimension of 1.
            image = np.squeeze(image)

        return image

    def _readStrippedContigImage(self, idx):
        """
        Read entire image where the orientation is stripped.
        """
        numstrips = lib.numberOfStrips(self.tfp)

        shape = self.h, self.w, self.spp
        dtype = self._determine_datatype()
        image = np.zeros(shape, dtype=dtype)

        stripshape = (self.rps, self.w, self.spp)

        for row in range(0, self.h, self.rps):
            rslice = slice(row, row + self.rps)

            stripnum = lib.computeStrip(self.tfp, row, 0)
            strip = np.zeros(stripshape, dtype=dtype)
            lib.readEncodedStrip(self.tfp, stripnum, strip)

            # Is it the last strip?  Is that last strip a full strip?
            # If not, then we need to shave off some rows.
            if stripnum == (numstrips - 1):
                if self.h % self.rps > 0:
                    strip = strip[:self.h % self.rps, :]

            image[rslice, :, :] = strip

        if self['SamplesPerPixel'] == 1:
            # squash the trailing dimension of 1.
            image = np.squeeze(image)

        return image

    def _readTiledImage(self, idx):
        """
        Helper routine for assembling an entire image out of tiles.
        """
        if self['PlanarConfig'] == lib.PlanarConfig.CONTIG:
            return self._readTiledContigImage(idx)
        else:
            return self._readTiledSeparateImage(idx)

    def _readTiledSeparateImage(self, idx):
        """
        Helper routine for assembling an entire image out of tiles.
        """
        numtilerows = int(round(self.h / self.th + 0.5))
        numtilecols = int(round(self.w / self.tw + 0.5))

        dtype = self._determine_datatype()
        shape = (self.h, self.w, self.spp)
        image = np.zeros(shape, dtype=dtype)

        tileshape = (self.th, self.tw)

        # Do the tile dimensions partition the image?  If not, then we will
        # need to chop up tiles read on the right hand side and on the bottom.
        partitioned = (self.h % self.th) == 0 and (self.w % self.th) == 0

        for row in range(0, self.h, self.th):
            rslice = slice(row, row + self.th)
            for col in range(0, self.w, self.tw):
                cslice = slice(col, col + self.tw)
                for plane in range(0, self.spp):
                    tilenum = lib.computeTile(self.tfp, col, row, 0, plane)

                    tile = np.zeros(tileshape, dtype=dtype)
                    lib.readEncodedTile(self.tfp, tilenum, tile)

                    if not partitioned and col // self.tw == numtilecols - 1:
                        # OK, we are on the right hand side.  Need to shave off
                        # some columns.
                        tile = tile[:, :(self.w - col)]
                    if not partitioned and row // self.th == numtilerows - 1:
                        # OK, we are on the bottom.  Need to shave off some
                        # rows.
                        tile = tile[:(self.h - row), :]

                    image[rslice, cslice, plane] = tile

        if self['SamplesPerPixel'] == 1:
            # squash the trailing dimension of 1.
            image = np.squeeze(image)

        return image

    def _readTiledContigImage(self, idx):
        """
        Helper routine for assembling an entire image out of tiles.
        """
        numtilerows = int(round(self.h / self.th + 0.5))
        numtilecols = int(round(self.w / self.tw + 0.5))

        shape = self.h, self.w, self.spp
        dtype = self._determine_datatype()
        image = np.zeros(shape, dtype=dtype)

        tileshape = (self.th, self.tw, self.spp)

        # Do the tile dimensions partition the image?  If not, then we will
        # need to chop up tiles read on the right hand side and on the bottom.
        partitioned = (self.h % self.th) == 0 and (self.w % self.th) == 0

        for row in range(0, self.h, self.th):
            rslice = slice(row, row + self.th)
            for col in range(0, self.w, self.tw):
                tilenum = lib.computeTile(self.tfp, col, row, 0, 0)
                cslice = slice(col, col + self.tw)

                tile = np.zeros(tileshape, dtype=dtype)
                lib.readEncodedTile(self.tfp, tilenum, tile)

                if not partitioned and col // self.tw == numtilecols - 1:
                    # OK, we are on the right hand side.  Need to shave off
                    # some columns.
                    tile = tile[:, :(self.w - col)]
                if not partitioned and row // self.th == numtilerows - 1:
                    # OK, we are on the bottom.  Need to shave off some rows.
                    tile = tile[:(self.h - row), :]

                image[rslice, cslice, :] = tile

        if self['SamplesPerPixel'] == 1:
            # squash the trailing dimension of 1.
            image = np.squeeze(image)

        return image

    def __getitem__(self, idx):
        """
        Either retrieve a named tag or read part/all of an image.
        """
        if isinstance(idx, slice):
            # Read the whole image?
            if not (('TileByteCounts' in self.tags.keys()) or
                    ('StripByteCounts' in self.tags.keys())):
                raise TIFFReadImageError('This IFD does not have an image')
            elif self.rgba:
                item = lib.readRGBAImageOriented(self.tfp, self.w, self.h)
            elif self['Compression'] == lib.Compression.OJPEG:
                # Force the issue with OJPEG.  This is the only case where we
                # just use RGBA mode without letting the user think about that.
                if idx.start is None and idx.stop is None and idx.step is None:
                    # case is [:]
                    item = lib.readRGBAImageOriented(self.tfp, self.w, self.h)
                    item = item[:, :, :3]
            elif lib.isTiled(self.tfp):
                item = self._readTiledImage(slice)
            else:
                item = self._readStrippedImage(slice)

        elif isinstance(idx, tuple):
            # Partial read?
            rowjslice = idx[0]
            colslice = idx[1]
            try:
                zslice = idx[2]
            except IndexError:
                zslice = None
            if lib.isTiled(self.tfp):
                image = self._readPartialTiled(rowslice, colslice, zslice)
            else:
                image = self._readPartialStripped(rowslice, colslice, zslice)

        elif isinstance(idx, str):
            if idx == 'JPEGColorMode':
                # This is a pseudo-tag that the user might not have set.
                item = lib.getFieldDefaulted(self.tfp, 'JPEGColorMode')
            elif idx in ['PlanarConfig', 'SampleFormat']:
                # These are tags that the user might not have set.
                item = lib.getFieldDefaulted(self.tfp, idx)
            else:
                item = self.tags[idx]

        return item

    def _readPartialStripped(self, rowslice, colslice, zslice):
        """
        Read a partial image according to the slice information.
        """
        if row_slice.start is None:
            starting_row = 0
        else:
            # We start assembling at the first row of the first strip, not
            # at the row the user specified.
            starting_row = (row_slice.start // self.rps) * self.rps

        if row_slice.stop is None:
            ending_row = self.h
        else:
            # Same with ending row.  Use the first row in the last strip that
            # we want.
            ending_row = (row_slice.stop // self.rps) * self.rps

        numstrips = lib.numberOfStrips(self.tfp)
        num_local_strips = (ending_row - starting_row) // self.rps + 1

        shape = num_local_strips * self.rps, self.w, self.spp
        dtype = self._determine_datatype()
        image = np.zeros(shape, dtype=dtype)

        stripshape = (self.rps, self.w, self.spp)
        strip = np.zeros(stripshape, dtype=dtype)

        # Assemble the strips.
        count = 0
        for row in range(starting_row, ending_row, self.rps):
            stripnum = lib.computeStrip(self.tfp, row, 0)
            lib.readEncodedStrip(self.tfp, stripnum, strip)

            # Figure out how to put the strip into the master image.
            image_row = row - starting_row
            rslice_image = slice(image_row, image_row + self.rps)
            image[rslice_image, colslice, :] = strip[:, colslice, :]

        # Is it the last strip?  Is that last strip a full strip?
        # If not, then we need to shave off some rows.
        if stripnum == (numstrips - 1):
            if self.h % self.rps > 0:
                strip = strip[:self.h % self.rps, :]

        if self['SamplesPerPixel'] == 1:
            # squash the trailing dimension of 1.
            image = np.squeeze(image)

        return image


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

            if tag_num == 306:
                # Datetime
                payload = dt.datetime.strptime(payload, '%Y:%m:%d %H:%M:%S')

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

        try:
            tag_name = self.tagnum2name[tag_num]
        except KeyError:
            tag_name = str(tag_num)

        # Special processing?
        if tag_num == 333:
            # InkNames
            payload = tuple(payload.split('\0'))
        elif tag_num == 42112:
            # GDAL_METADATA is an XML fragment.
            payload = etree.fromstring(payload)

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
