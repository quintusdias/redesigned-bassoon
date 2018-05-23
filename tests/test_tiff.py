# Local imports
import itertools
import pathlib
import tempfile
import unittest

# Third party library imports
import numpy as np
import skimage.data
import skimage.measure

# Local imports
from tiffany.tiffany import TIFF, JPEGColorModeRawError
from tiffany.lib import (
    Compression, JPEGProc, Photometric, PlanarConfig, JPEGColorMode,
    ResolutionUnit
)


class TestSuite(unittest.TestCase):

    def _get_path(self, filename):
        """
        Return full path of a test file.
        """
        directory = pathlib.Path(__file__).parent
        return directory / 'data' / filename

    def test_read_ojpeg(self):
        """
        Scenario: Read a TIFF with OJPEG compression.

        Expected Result:  The tags match the output of TIFFDUMP.  The image
        size matches the tag values.
        """
        path = self._get_path('zackthecat.tif')

        t = TIFF(path)
        image = t[:]

        self.assertEqual(image.shape, (t.h, t.w, t.spp))

        self.assertEqual(t['bitspersample'], (8, 8, 8))
        self.assertEqual(t['compression'], Compression.OJPEG)
        self.assertEqual(t['photometric'], Photometric.YCBCR)
        self.assertEqual(t['xresolution'], 75.0)
        self.assertEqual(t['yresolution'], 75.0)
        self.assertEqual(t['planarconfig'], PlanarConfig.CONTIG)
        self.assertEqual(t['resolutionunit'], ResolutionUnit.INCH)
        self.assertEqual(t['tilewidth'], 240)
        self.assertEqual(t['tilelength'], 224)
        self.assertEqual(t['jpegproc'], JPEGProc.BASELINE)

        self.assertEqual(t['jpegqtables'], (7364, 7428, 7492))
        self.assertEqual(t['jpegdctables'], (7568, 7596, 7624))
        self.assertEqual(t['jpegactables'], (7664, 7842, 8020))

        np.testing.assert_allclose(t['ycbcrcoefficients'],
                                   (0.299, 0.587, 0.114))

        self.assertEqual(t['ycbcrsubsampling'], (2, 2))
        self.assertEqual(t['referenceblackwhite'],
                         (16, 235, 128, 240, 128, 240))

    def test_write_read_ycbcr_jpeg_rgb(self):
        """
        Scenario: Write the scikit-image "astronaut" as ycbcr/jpeg.

        Expected Result:  The image should be lossy to some degree.
        """
        expected = skimage.data.astronaut()

        photometrics = (Photometric.YCBCR,)
        compressions = (Compression.JPEG,)
        planars = (PlanarConfig.CONTIG,)
        subsamplings = ((1, 1), (1, 2), (2, 1), (2, 2))
        tiled = (True, False)
        modes = ('w', 'w8')

        g = itertools.product(
            photometrics, compressions, planars, tiled, subsamplings, modes
        )
        for photometric, compression, pc, tiled, subsampling, mode in g:
            with self.subTest(photometric=photometric,
                              compression=compression,
                              planar_config=pc,
                              tiled=tiled,
                              subsampling=subsampling,
                              mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    t['photometric'] = photometric
                    t['compression'] = compression
                    t['jpegcolormode'] = JPEGColorMode.RGB

                    w, h, nz = expected.shape
                    t['imagewidth'] = expected.shape[1]
                    t['imagelength'] = expected.shape[0]
                    t['planarconfig'] = pc
                    t['ycbcrsubsampling'] = subsampling

                    t['bitspersample'] = 8
                    t['samplesperpixel'] = 3
                    if tiled:
                        tw, th = int(w / 2), int(h / 2)
                        t['tilelength'] = th
                        t['tilewidth'] = tw
                    else:
                        rps = int(expected.shape[0] / 2)
                        t['rowsperstrip'] = rps

                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(t['photometric'], photometric)
                    self.assertEqual(t['planarconfig'], pc)
                    self.assertEqual(t['imagewidth'], w)
                    self.assertEqual(t['imagelength'], h)
                    self.assertEqual(t['bitspersample'], (8, 8, 8))
                    self.assertEqual(t['samplesperpixel'], 3)
                    if tiled:
                        self.assertEqual(t['tilewidth'], tw)
                        self.assertEqual(t['tilelength'], th)
                    else:
                        self.assertEqual(t['rowsperstrip'], rps)
                    self.assertEqual(t['compression'], compression)

                    actual = t[:]

                metric = skimage.measure.compare_psnr(expected, actual)
                self.assertTrue(metric > 5)

    def test_write_read_ycbcr_jpeg_raw(self):
        """
        Scenario: Write the scikit-image "astronaut" as ycbcr/jpeg.

        Expected Result:  A runtime error should be raised.  Writing YCbCr/JPEG
        with the JPEGColorMode as RAW is not a good workflow here.  One should
        use RGB instead.
        """
        expected = skimage.data.astronaut()
        ycbcr = skimage.color.rgb2ycbcr(expected).astype(np.uint8)

        photometrics = (Photometric.YCBCR,)
        compressions = (Compression.JPEG,)
        planars = (PlanarConfig.CONTIG,)
        tiled = (True, False)
        modes = ('w', 'w8')

        g = itertools.product(
            photometrics, compressions, planars, tiled, modes
        )
        for photometric, compression, planar_config, tiled, mode in g:
            with self.subTest(photometric=photometric,
                              compression=compression,
                              planar_config=planar_config,
                              tiled=tiled,
                              mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    t['photometric'] = photometric

                    w, h, nz = expected.shape
                    t['imagewidth'] = expected.shape[1]
                    t['imagelength'] = expected.shape[0]
                    t['planarconfig'] = planar_config
                    t['ycbcrsubsampling'] = 1, 1

                    t['bitspersample'] = 8
                    t['samplesperpixel'] = 3
                    if tiled:
                        tw, th = int(w / 2), int(h / 2)
                        t['tilelength'] = th
                        t['tilewidth'] = tw
                    else:
                        rps = int(expected.shape[0] / 2)
                        t['rowsperstrip'] = rps
                    t['compression'] = compression

                    with self.assertRaises(JPEGColorModeRawError):
                        t[:] = ycbcr

    def test_write_read_rgb_jpeg(self):
        """
        Scenario: Write the scikit-image "astronaut" as rgb/jpeg.

        Expected Result:  The image should be lossy to some degree.
        """
        expected = skimage.data.astronaut()

        photometrics = (Photometric.RGB,)
        compressions = (Compression.JPEG,)
        planars = (PlanarConfig.CONTIG,)
        tiled = (True, False)
        modes = ('w', 'w8')

        g = itertools.product(
            photometrics, compressions, planars, tiled, modes
        )
        for photometric, compression, planar_config, tiled, mode in g:
            with self.subTest(photometric=photometric,
                              compression=compression,
                              planar_config=planar_config,
                              tiled=tiled,
                              mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    t['photometric'] = photometric

                    w, h, nz = expected.shape
                    t['imagewidth'] = expected.shape[1]
                    t['imagelength'] = expected.shape[0]
                    t['planarconfig'] = planar_config

                    t['bitspersample'] = 8
                    t['samplesperpixel'] = 3
                    if tiled:
                        tw, th = int(w / 2), int(h / 2)
                        t['tilelength'] = th
                        t['tilewidth'] = tw
                    else:
                        rps = int(expected.shape[0] / 2)
                        t['rowsperstrip'] = rps
                    t['compression'] = compression

                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(t['photometric'], photometric)
                    self.assertEqual(t['planarconfig'], planar_config)
                    self.assertEqual(t['imagewidth'], w)
                    self.assertEqual(t['imagelength'], h)
                    self.assertEqual(t['bitspersample'], (8, 8, 8))
                    self.assertEqual(t['samplesperpixel'], 3)
                    if tiled:
                        self.assertEqual(t['tilewidth'], tw)
                        self.assertEqual(t['tilelength'], th)
                    else:
                        self.assertEqual(t['rowsperstrip'], rps)
                    self.assertEqual(t['compression'], compression)

                    actual = t[:]

                metric = skimage.measure.compare_psnr(expected, actual)
                self.assertTrue(metric > 37)

    def test_write_read_rgb(self):
        """
        Scenario: Write the scikit-image "astronaut" to file.

        Expected Result:  The image should be lossless for the appropriate
        compression schemes.
        """
        expected = skimage.data.astronaut()

        photometrics = (Photometric.RGB,)
        compressions = (Compression.NONE, Compression.LZW,
                        Compression.PACKBITS, Compression.DEFLATE,
                        Compression.ADOBE_DEFLATE, Compression.LZMA)
        planars = (PlanarConfig.CONTIG,)
        tiled = (True, False)
        modes = ('w', 'w8')

        g = itertools.product(
            photometrics, compressions, planars, tiled, modes
        )
        for photometric, compression, planar_config, tiled, mode in g:
            with self.subTest(photometric=photometric,
                              compression=compression,
                              planar_config=planar_config,
                              tiled=tiled,
                              mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    t['photometric'] = photometric

                    w, h, nz = expected.shape
                    t['imagewidth'] = expected.shape[1]
                    t['imagelength'] = expected.shape[0]
                    t['planarconfig'] = planar_config

                    t['bitspersample'] = 8
                    t['samplesperpixel'] = 3
                    if tiled:
                        tw, th = int(w / 2), int(h / 2)
                        t['tilelength'] = th
                        t['tilewidth'] = tw
                    else:
                        rps = int(expected.shape[0] / 2)
                        t['rowsperstrip'] = rps
                    t['compression'] = compression
                    t['resolutionunit'] = ResolutionUnit.INCH
                    t['xresolution'] = 7.5
                    t['yresolution'] = 7.5

                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(t['photometric'], photometric)
                    self.assertEqual(t['planarconfig'], planar_config)
                    self.assertEqual(t['imagewidth'], w)
                    self.assertEqual(t['imagelength'], h)
                    self.assertEqual(t['bitspersample'], (8, 8, 8))
                    self.assertEqual(t['samplesperpixel'], 3)
                    if tiled:
                        self.assertEqual(t['tilewidth'], tw)
                        self.assertEqual(t['tilelength'], th)
                    else:
                        self.assertEqual(t['rowsperstrip'], rps)
                    self.assertEqual(t['compression'], compression)

                    self.assertEqual(t['resolutionunit'], ResolutionUnit.INCH)
                    self.assertEqual(t['xresolution'], 7.5)
                    self.assertEqual(t['yresolution'], 7.5)

                    actual = t[:]

                np.testing.assert_equal(actual, expected)

    def test_write_read_greyscale(self):
        """
        Scenario: Write the scikit-image "camera" to file.

        Expected Result:  The data should be round-trip the same for greyscale
        photometric interpretations and lossless compression schemes.
        """
        expected = skimage.data.camera()

        photometrics = (Photometric.MINISBLACK, Photometric.MINISWHITE)
        compressions = (Compression.NONE, Compression.LZW,
                        Compression.PACKBITS, Compression.DEFLATE,
                        Compression.ADOBE_DEFLATE, Compression.LZMA)
        tiled = (True, False)
        modes = ('w', 'w8')

        g = itertools.product(photometrics, compressions, tiled, modes)
        for photometric, compression, tiled, mode in g:
            with self.subTest(photometric=photometric,
                              compression=compression,
                              tiled=tiled,
                              mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    t['photometric'] = photometric

                    w, h = expected.shape
                    t['imagewidth'] = expected.shape[1]
                    t['imagelength'] = expected.shape[0]

                    t['bitspersample'] = 8
                    t['samplesperpixel'] = 1
                    if tiled:
                        tw, th = int(w / 2), int(h / 2)
                        t['tilelength'] = th
                        t['tilewidth'] = tw
                    else:
                        rps = int(expected.shape[0] / 2)
                        t['rowsperstrip'] = rps
                    t['compression'] = compression

                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(t['photometric'], photometric)
                    self.assertEqual(t['imagewidth'], w)
                    self.assertEqual(t['imagelength'], h)
                    self.assertEqual(t['bitspersample'], 8)
                    self.assertEqual(t['samplesperpixel'], 1)
                    if tiled:
                        self.assertEqual(t['tilewidth'], tw)
                        self.assertEqual(t['tilelength'], th)
                    else:
                        self.assertEqual(t['rowsperstrip'], rps)
                    self.assertEqual(t['compression'], compression)

                    actual = t[:]

                np.testing.assert_equal(actual, expected)

    def test_write_read_unequally_partitioned_grayscale_images(self):
        """
        Scenario: Write the scikit-image "camera" to file.  The tiles and
        strips do not equally partition the image.

        Expected Result:  The data should be round-trip the same for greyscale
        photometric interpretations and lossless compression schemes.
        """
        expected = skimage.data.camera()

        for tiled in True, False:
            with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                t = TIFF(tfile.name, mode='w')
                t['photometric'] = Photometric.MINISBLACK

                w, h = expected.shape
                t['imagewidth'] = expected.shape[1]
                t['imagelength'] = expected.shape[0]

                t['bitspersample'] = 8
                t['samplesperpixel'] = 1
                if tiled:
                    tw, th = 160, 160
                    t['tilelength'] = th
                    t['tilewidth'] = tw
                else:
                    rps = 160
                    t['rowsperstrip'] = rps
                t['compression'] = Compression.NONE

                t[:] = expected

                del t

                t = TIFF(tfile.name)
                self.assertEqual(t['photometric'], Photometric.MINISBLACK)
                self.assertEqual(t['imagewidth'], w)
                self.assertEqual(t['imagelength'], h)
                self.assertEqual(t['bitspersample'], 8)
                self.assertEqual(t['samplesperpixel'], 1)
                if tiled:
                    self.assertEqual(t['tilewidth'], tw)
                    self.assertEqual(t['tilelength'], th)
                else:
                    self.assertEqual(t['rowsperstrip'], rps)
                self.assertEqual(t['compression'], Compression.NONE)

                actual = t[:]

            np.testing.assert_equal(actual, expected)

    def test_write_read_unequally_partitioned_rgb_images(self):
        """
        Scenario: Write the scikit-image RGB "astronaut" image to file.
        The tiles and strips do not equally partition the image.

        Expected Result:  The data should be round-trip the same for RGB
        photometric interpretations and lossless compression schemes.
        """
        expected = skimage.data.astronaut()

        for tiled in True, False:
            with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                t = TIFF(tfile.name, mode='w')
                t['photometric'] = Photometric.RGB

                w, h, spp = expected.shape
                t['imagewidth'] = w
                t['imagelength'] = h
                t['samplesperpixel'] = spp
                t['planarconfig'] = PlanarConfig.CONTIG

                t['bitspersample'] = 8
                if tiled:
                    tw, th = 160, 160
                    t['tilelength'] = th
                    t['tilewidth'] = tw
                else:
                    rps = 160
                    t['rowsperstrip'] = rps
                t['compression'] = Compression.NONE

                t[:] = expected

                del t

                t = TIFF(tfile.name)
                self.assertEqual(t['photometric'], Photometric.RGB)
                self.assertEqual(t['imagewidth'], w)
                self.assertEqual(t['imagelength'], h)
                self.assertEqual(t['bitspersample'], (8, 8, 8))
                self.assertEqual(t['samplesperpixel'], 3)
                if tiled:
                    self.assertEqual(t['tilewidth'], tw)
                    self.assertEqual(t['tilelength'], th)
                else:
                    self.assertEqual(t['rowsperstrip'], rps)
                self.assertEqual(t['compression'], Compression.NONE)

                actual = t[:]

            np.testing.assert_equal(actual, expected)
