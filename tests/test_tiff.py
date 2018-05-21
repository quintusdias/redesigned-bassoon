# Local imports
import itertools
import tempfile
import unittest

# Third party library imports
import numpy as np
import skimage.data
import skimage.measure

# Local imports
from tiffany.tiffany import TIFF
from tiffany.lib import Compression, Photometric, PlanarConfig, JPEGColorMode


class TestSuite(unittest.TestCase):

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
        for photometric, compression, planar_config, tiled, subsampling, mode in g:
            with self.subTest(photometric=photometric,
                              compression=compression,
                              planar_config=planar_config,
                              tiled=tiled,
                              subsampling=subsampling,
                              mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    t['photometric'] = photometric

                    w, h, nz = expected.shape
                    t['imagewidth'] = expected.shape[1]
                    t['imagelength'] = expected.shape[0]
                    t['planarconfig'] = planar_config
                    t['jpegcolormode'] = JPEGColorMode.RGB
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
                    t['compression'] = compression

                    t[:] = ycbcr

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

                metric = skimage.measure.compare_psnr(ycbcr, actual)
                self.assertTrue(metric > 40)

    def test_write_read_ycbcr_jpeg_raw(self):
        """
        Scenario: Write the scikit-image "astronaut" as ycbcr/jpeg.

        Expected Result:  The image should be lossy to some degree.
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

                    t[:] = ycbcr

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

                metric = skimage.measure.compare_psnr(ycbcr, actual)
                self.assertTrue(metric > 40)

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
