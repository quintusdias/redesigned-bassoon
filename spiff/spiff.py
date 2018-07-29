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
    def pc(self):
        """
        Shortcut for the planar configuration.
        """
        return self['PlanarConfig']

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

    def write_directory(self):
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

    def set_subdirectory(self, offset):
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

            try:
                if (((self['Photometric'] == lib.Photometric.YCBCR) and
                     (idx == 'Compression') and
                     (value != lib.Compression.JPEG))):
                    msg = (
                        "YCbCr photometric interpretation requires JPEG "
                        "compression."
                    )
                    raise TypeError(msg)
            except KeyError:
                pass

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
                item = self._readTiledImage(idx)
            else:
                item = self._readStrippedImage(idx)

        elif isinstance(idx, tuple):
            # Partial read?
            rowslice = idx[0]
            colslice = idx[1]
            try:
                zslice = idx[2]
            except IndexError:
                zslice = None
            if lib.isTiled(self.tfp):
                item = self._readPartialTiled(rowslice, colslice, zslice)
            else:
                item = self._readPartialStripped(rowslice, colslice, zslice)

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
        Read a partial stripped image according to the slice information.
        """
        # Determine the upper left pixel of the starting strip.
        if rowslice.start is None:
            ul_sy = 0
            rowslice = slice(0, rowslice.stop, rowslice.step)
        else:
            ul_sy = (rowslice.start // self.rps) * self.rps

        if colslice.start is None:
            colslice = slice(0, colslice.stop, colslice.step)

        ul_stripnum = lib.computeStrip(self.tfp, ul_sy, 0)

        # Determine the upper left pixel of the leftmost tile in the starting
        # tile column.  The wording is funny, we'll refer to it as lower left
        # row and column.
        if rowslice.stop is None:
            # Go to the end.
            ll_sy = self.h - self.rps + 1
        else:
            ll_sy = min(self.h, rowslice.stop)
        # Make it at least one strip down.
        ll_sy = max(ll_sy, ul_sy + self.rps)

        # Are the requested extents bigger than the image?  If so, roll them
        # back.
        if rowslice.stop > self.h:
            rowslice = slice(rowslice.start, self.h, rowslice.step)
        if colslice.stop > self.w:
            colslice = slice(colslice.start, self.w, colslice.step)

        # Initialize the returned image
        numrows = rowslice.stop - rowslice.start
        numcols = colslice.stop - colslice.start
        shape = (numrows, numcols, self.spp)
        dtype = self._determine_datatype()
        image = np.zeros(shape, dtype=dtype)

        # Initialize the tile.
        if self.pc == lib.PlanarConfig.SEPARATE:
            strip = np.zeros((self.rps, self.w, 1), dtype=dtype)
        else:
            strip = np.zeros((self.rps, self.w, self.spp), dtype=dtype)

        # Compute some strip extents.
        ul_stripnum = lib.computeStrip(self.tfp, ul_sy, 0)
        ll_stripnum = lib.computeStrip(self.tfp, rowslice.stop, 0)

        for row in range(ul_sy, ll_sy, self.rps):

            # Determine the source row slice for this strip
            if row == ul_sy:
                # First tile row.
                sy1 = rowslice.start % self.rps

                # Is the end extent inside that first tile?  If so, then
                # the end extent sx2 is computed the same as sx1.
                if ul_stripnum == ll_stripnum:
                    sy2 = rowslice.stop % self.rps
                else:
                    sy2 = self.rps

            elif (ll_sy - row) < self.rps:
                # Last tile row.
                sy1 = 0
                sy2 = rowslice.stop % self.rps
            else:
                # Interior tile row.
                sy1 = 0
                sy2 = self.rps
            syslice = slice(sy1, sy2)

            # Determine the destination row slice for this tile row.
            if row == ul_sy:
                # First strip
                dy1 = 0
                dy2 = sy2 - sy1
            elif (ll_sy - row) < self.rps:
                # Last strip
                dy1 = dy2
                dy2 = image.shape[0]
            else:
                # Interior strip.
                dy1 = dy2
                dy2 = dy2 + self.rps
            dyslice = slice(dy1, dy2)

            if self.pc == lib.PlanarConfig.SEPARATE:
                for sample in range(0, self.spp):
                    stripnum = lib.computeStrip(self.tfp, row, sample)
                    lib.readEncodedStrip(self.tfp, stripnum, strip)
                    image[dyslice, :, sample] = strip[syslice, colslice].squeeze()  # noqa: E501
            else:
                stripnum = lib.computeStrip(self.tfp, row, 0)
                lib.readEncodedStrip(self.tfp, stripnum, strip)
                image[dyslice, :, :] = strip[syslice, colslice, :]

        return image

    def _readPartialTiled(self, rowslice, colslice, zslice):
        """
        Read a partial tiled separate image according to the slice
        information.
        """
        # Determine the upper left pixel of the starting tile.
        if rowslice.start is None:
            ul_ty = 0
            rowslice = slice(0, rowslice.stop, rowslice.step)
        else:
            ul_ty = (rowslice.start // self.th) * self.th

        if colslice.start is None:
            ul_tx = 0
            colslice = slice(0, colslice.stop, colslice.step)
        else:
            ul_tx = (colslice.start // self.tw) * self.tw

        ul_tilenum = lib.computeTile(self.tfp, ul_tx, ul_ty, 0, 0)

        # Determine the upper left pixel of the rightmost tile in the starting
        # tile row.  The wording is funny, we'll refer to it as upper right
        # row and column.
        ur_ty = ul_ty
        if colslice.stop is None:
            # Go to the end.
            ur_tx = self.w - self.tw + 1
        else:
            ur_tx = min(self.w, colslice.stop)

        # Make it at least one tile over.
        ur_tx = max(ur_tx, ul_tx + self.tw)
        ur_tilenum = lib.computeTile(self.tfp, ur_tx, ur_ty, 0, 0)

        # Determine the upper left pixel of the leftmost tile in the starting
        # tile column.  The wording is funny, we'll refer to it as lower left
        # row and column.
        if rowslice.stop is None:
            # Go to the end.
            ll_ty = self.h - self.th + 1
        else:
            ll_ty = min(self.h, rowslice.stop)
        # Make it at least one tile down.
        ll_ty = max(ll_ty, ul_ty + self.th)

        # Determine the upper left pixel of the rightmost tile in the ending
        # tile column.  The wording is funny, we'll refer to it as lower right
        # row and column.
        lr_ty = ll_ty

        # Are the requested extents bigger than the image?  If so, roll them
        # back.
        if rowslice.stop > self.h:
            rowslice = slice(rowslice.start, self.h, rowslice.step)
        if colslice.stop > self.w:
            colslice = slice(colslice.start, self.w, colslice.step)

        # Initialize the returned image
        numrows = rowslice.stop - rowslice.start
        numcols = colslice.stop - colslice.start
        shape = (numrows, numcols, self.spp)
        dtype = self._determine_datatype()
        image = np.zeros(shape, dtype=dtype)

        # Initialize the tile.
        if self.pc == lib.PlanarConfig.SEPARATE:
            tile = np.zeros((self.th, self.tw, 1), dtype=dtype)
        else:
            tile = np.zeros((self.th, self.tw, self.spp), dtype=dtype)

        # Compute some tile extents.
        ul_tilenum = lib.computeTile(self.tfp, ul_tx, ul_ty, 0, 0)
        ur_tilenum = lib.computeTile(self.tfp, colslice.stop, ul_ty, 0, 0)
        ll_tilenum = lib.computeTile(self.tfp, ul_tx, rowslice.stop, 0, 0)

        for tx in range(ul_tx, ur_tx, self.tw):

            # Determine the source column slice for this tile column.
            if tx == ul_tx:
                # First tile column.
                sx1 = colslice.start % self.tw

                # Is the end extent inside that first tile?  If so, then the
                # end extent sx2 is computed the same as sx1.
                if ul_tilenum == ur_tilenum:
                    sx2 = colslice.stop % self.tw
                else:
                    sx2 = self.tw

            elif (ur_tx - tx) < self.tw:
                # Last tile column.
                sx1 = 0
                sx2 = colslice.stop % self.tw
            else:
                # Interior tile column.
                sx1 = 0
                sx2 = self.tw
            sxslice = slice(sx1, sx2)

            # Determine the destination column slice for this tile column.
            if tx == ul_tx:
                # First tile column.
                dx1 = 0
                dx2 = sx2 - sx1
            elif (ur_tx - tx) < self.tw:
                # Last tile column.
                dx1 = dx2
                dx2 = image.shape[1]
            else:
                # Interior tile column.
                dx1 = dx2
                dx2 = dx2 + self.tw
            dxslice = slice(dx1, dx2)

            for ty in range(ul_ty, lr_ty, self.th):

                # Determine the source row slice for this tile row.
                if ty == ul_ty:
                    # First tile row.
                    sy1 = rowslice.start % self.th

                    # Is the end extent inside that first tile?  If so, then
                    # the end extent sx2 is computed the same as sx1.
                    if ul_tilenum == ll_tilenum:
                        sy2 = rowslice.stop % self.th
                    else:
                        sy2 = self.th

                elif (ll_ty - ty) < self.th:
                    # Last tile row.
                    sy1 = 0
                    sy2 = rowslice.stop % self.th
                else:
                    # Interior tile row.
                    sy1 = 0
                    sy2 = self.th
                syslice = slice(sy1, sy2)

                # Determine the destination row slice for this tile row.
                if ty == ul_ty:
                    # First tile row.
                    dy1 = 0
                    dy2 = sy2 - sy1
                elif (ll_ty - ty) < self.th:
                    # Last tile row.
                    dy1 = dy2
                    dy2 = image.shape[0]
                else:
                    # Interior tile row.
                    dy1 = dy2
                    dy2 = dy2 + self.th
                dyslice = slice(dy1, dy2)

                if self.pc == lib.PlanarConfig.SEPARATE:
                    for sample in range(0, self.spp):
                        tilenum = lib.computeTile(self.tfp, tx, ty, 0, sample)
                        lib.readEncodedTile(self.tfp, tilenum, tile)
                        image[dyslice, dxslice, sample] = tile[syslice, sxslice].squeeze()  # noqa: E501
                else:
                    tilenum = lib.computeTile(self.tfp, tx, ty, 0, 0)
                    lib.readEncodedTile(self.tfp, tilenum, tile)
                    image[dyslice, dxslice, :] = tile[syslice, sxslice, :]

        return image

    def _readPartialStrippedContig(self, rowslice, colslice, zslice):
        """
        Read a partial stripped planar configuous image according to the slice
        information.
        """
        starting_row = 0 if rowslice.start is None else rowslice.start
        starting_strip = lib.computeStrip(self.tfp, starting_row, 0)

        if rowslice.stop is None:
            # Go to the end.
            ending_row = self.h - self.rps + 1
        else:
            ending_row = rowslice.stop
        ending_strip = lib.computeStrip(self.tfp, ending_row - 1, 0)

        starting_col = 0 if colslice.start is None else colslice.start
        ending_col = self.w if colslice.stop is None else colslice.stop

        # Initialize the returned image.
        shape = ending_row - starting_row, ending_col - starting_col, self.spp
        dtype = self._determine_datatype()
        image = np.zeros(shape, dtype=dtype)

        # Initialize the strip image.  We reuse this each time we read a strip.
        stripshape = (self.rps, self.w, self.spp)
        strip = np.zeros(stripshape, dtype=dtype)

        # Assemble the strips.
        # There are three cases, the first strip, the last strip, and all the
        # interior strips.

        # 1st strip
        lib.readEncodedStrip(self.tfp, starting_strip, strip)
        src_r_slice = slice(starting_row % self.rps,
                            max(self.rps, ending_row % self.rps))
        dest_r_slice = slice(0, src_r_slice.stop - src_r_slice.start)
        image[dest_r_slice, :, :] = strip[src_r_slice, colslice, :]

        # All the interior strips.
        src_r_slice = slice(0, self.rps)
        for stripnum in range(starting_strip + 1, ending_strip):
            lib.readEncodedStrip(self.tfp, stripnum, strip)

            # Extract the partial image from the strip.
            dest_r_slice = slice(dest_r_slice.stop,
                                 dest_r_slice.stop + self.rps)
            image[dest_r_slice, :, :] = strip[src_r_slice, colslice, :]

        # And do the last strip
        if ending_strip > starting_strip:
            lib.readEncodedStrip(self.tfp, ending_strip, strip)

            src_r_slice = slice(0, rowslice.stop % self.rps)
            dest_r_slice = slice(dest_r_slice.stop,
                                 dest_r_slice.stop + src_r_slice.stop)

            image[dest_r_slice, :, :] = strip[src_r_slice, colslice, :]

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
