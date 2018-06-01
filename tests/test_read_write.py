# Local imports
import datetime
import itertools
import pathlib
import platform
import tempfile
import unittest

# Third party library imports
import numpy as np
import skimage.data
import skimage.measure

# Local imports
from tiffany.tiffany import TIFF, JPEGColorModeRawError
from tiffany import lib


@unittest.skipIf(platform.system() == 'Windows', "tempfile issue on Windows")
class TestSuite(unittest.TestCase):

    def _get_path(self, filename):
        """
        Return full path of a test file.
        """
        directory = pathlib.Path(__file__).parent
        return directory / 'data' / filename

    def test_write_read_separated_non_cmyk(self):
        """
        Scenario: Write the scikit-image "astronaut" as separated with a
        non-cmyk inkset.

        Expected Result:  InkSet, NumberOfInks, and InkNames should validate.
        """
        expected = skimage.data.astronaut()

        photo = lib.Photometric.SEPARATED
        comp = lib.Compression.NONE
        pc = lib.PlanarConfig.CONTIG

        tiled = (True, False)
        modes = ('w', 'w8')

        g = itertools.product(tiled, modes)
        for tiled, mode in g:
            with self.subTest(tiled=tiled, mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:

                    t = TIFF(tfile.name, mode=mode)
                    t['Photometric'] = photo
                    t['Compression'] = comp
                    t['BitsPerSample'] = 8
                    t['SamplesPerPixel'] = 3
                    t['PlanarConfig'] = pc

                    h, w, nz = expected.shape
                    t['ImageWidth'] = w
                    t['ImageLength'] = h

                    if tiled:
                        tw, th = 256, 256
                        t['TileLength'] = th
                        t['TileWidth'] = tw
                    else:
                        rps = 256
                        t['RowsPerStrip'] = rps

                    t['InkSet'] = lib.InkSet.MULTIINK
                    t['NumberOfInks'] = 3
                    t['InkNames'] = ('R', 'G', 'B')

                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(t['Photometric'], photo)
                    self.assertEqual(t['Compression'], comp)
                    self.assertEqual(t['ImageWidth'], w)
                    self.assertEqual(t['ImageLength'], h)
                    self.assertEqual(t['SamplesPerPixel'], 3)
                    if tiled:
                        self.assertEqual(t['TileWidth'], tw)
                        self.assertEqual(t['TileLength'], th)
                    else:
                        self.assertEqual(t['RowsPerStrip'], rps)
                    self.assertEqual(t['InkSet'], lib.InkSet.MULTIINK)
                    self.assertEqual(t['NumberOfInks'], 3)
                    self.assertEqual(t['InkNames'], ('R', 'G', 'B'))

                    actual = t[:]

                np.testing.assert_array_equal(actual, expected)

    def test_write_read_floating_point_minisblack(self):
        """
        Scenario: Write the scikit-image "stereo_motorcycle" as minisblack and
        floating point.

        Expected Result:  The image should be lossless.
        """
        _, _, motorcycle = skimage.data.stereo_motorcycle()

        photo = lib.Photometric.MINISBLACK
        comp = lib.Compression.NONE
        sf = lib.SampleFormat.IEEEFP

        tiled = (True, False)
        modes = ('w', 'w8')
        bpss = (32, 64)

        g = itertools.product(tiled, modes, bpss)
        for tiled, mode, bps in g:
            with self.subTest(tiled=tiled, mode=mode, bps=bps):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:

                    if bps == 64:
                        expected = motorcycle.astype(np.float64)
                    else:
                        expected = motorcycle.astype(np.float32)

                    t = TIFF(tfile.name, mode=mode)
                    t['Photometric'] = photo
                    t['Compression'] = comp
                    t['SampleFormat'] = sf
                    t['BitsPerSample'] = bps
                    t['SamplesPerPixel'] = 1

                    h, w = expected.shape
                    t['ImageWidth'] = expected.shape[1]
                    t['ImageLength'] = expected.shape[0]

                    if tiled:
                        tw, th = 256, 256
                        t['TileLength'] = th
                        t['TileWidth'] = tw
                    else:
                        rps = 256
                        t['RowsPerStrip'] = rps

                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(t['Photometric'], photo)
                    self.assertEqual(t['Compression'], comp)
                    self.assertEqual(t['SampleFormat'], sf)
                    self.assertEqual(t['BitsPerSample'], bps)
                    self.assertEqual(t['ImageWidth'], w)
                    self.assertEqual(t['ImageLength'], h)
                    self.assertEqual(t['SamplesPerPixel'], 1)
                    if tiled:
                        self.assertEqual(t['TileWidth'], tw)
                        self.assertEqual(t['TileLength'], th)
                    else:
                        self.assertEqual(t['RowsPerStrip'], rps)

                    actual = t[:]

                np.testing.assert_array_equal(actual, expected)

    def test_write_read_ycbcr_jpeg_rgb(self):
        """
        Scenario: Write the scikit-image "astronaut" as ycbcr/jpeg.

        Expected Result:  The image should be lossy to some degree.
        """
        expected = skimage.data.astronaut()

        photometrics = (lib.Photometric.YCBCR,)
        Compressions = (lib.Compression.JPEG,)
        planars = (lib.PlanarConfig.CONTIG,)
        subsamplings = ((1, 1), (1, 2), (2, 1), (2, 2))
        tiled = (True, False)
        modes = ('w', 'w8')

        g = itertools.product(
            photometrics, Compressions, planars, tiled, subsamplings, modes
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
                    t['Photometric'] = photometric
                    t['Compression'] = compression
                    t['JPEGColorMode'] = lib.JPEGColorMode.RGB

                    w, h, nz = expected.shape
                    t['ImageWidth'] = expected.shape[1]
                    t['ImageLength'] = expected.shape[0]
                    t['PlanarConfig'] = pc
                    t['YCbCrSubsampling'] = subsampling

                    t['BitsPerSample'] = 8
                    t['SamplesPerPixel'] = 3
                    if tiled:
                        tw, th = int(w / 2), int(h / 2)
                        t['TileLength'] = th
                        t['TileWidth'] = tw
                    else:
                        rps = int(expected.shape[0] / 2)
                        t['RowsPerStrip'] = rps

                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(t['Photometric'], photometric)
                    self.assertEqual(t['PlanarConfig'], pc)
                    self.assertEqual(t['Compression'], compression)
                    self.assertEqual(t['ImageWidth'], w)
                    self.assertEqual(t['ImageLength'], h)
                    self.assertEqual(t['BitsPerSample'], (8, 8, 8))
                    self.assertEqual(t['SamplesPerPixel'], 3)
                    if tiled:
                        self.assertEqual(t['TileWidth'], tw)
                        self.assertEqual(t['TileLength'], th)
                    else:
                        self.assertEqual(t['RowsPerStrip'], rps)

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

        photometrics = (lib.Photometric.YCBCR,)
        Compressions = (lib.Compression.JPEG,)
        planars = (lib.PlanarConfig.CONTIG,)
        tiled = (True, False)
        modes = ('w', 'w8')

        g = itertools.product(
            photometrics, Compressions, planars, tiled, modes
        )
        for photometric, compression, planar_config, tiled, mode in g:
            with self.subTest(photometric=photometric,
                              Compression=compression,
                              planar_config=planar_config,
                              tiled=tiled,
                              mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    t['Photometric'] = photometric

                    w, h, nz = expected.shape
                    t['ImageWidth'] = expected.shape[1]
                    t['ImageLength'] = expected.shape[0]
                    t['PlanarConfig'] = planar_config
                    t['YCbCrSubsampling'] = 1, 1

                    t['BitsPerSample'] = 8
                    t['SamplesPerPixel'] = 3
                    if tiled:
                        tw, th = int(w / 2), int(h / 2)
                        t['TileLength'] = th
                        t['TileWidth'] = tw
                    else:
                        rps = int(expected.shape[0] / 2)
                        t['RowsPerStrip'] = rps
                    t['Compression'] = compression

                    with self.assertRaises(JPEGColorModeRawError):
                        t[:] = ycbcr

    def test_write_read_two_alpha_on_grayscale(self):
        """
        Scenario: Write the scikit-image "camera" with two alpha layers.

        Expected Result:  The image should be lossless.  We confirm grayscale
        with two alpha layers.
        """
        gray = skimage.data.camera().reshape((512, 512, 1))

        # Create a gradient alpha layer.
        x = np.arange(0, 256, 0.5).astype(np.uint8).reshape(512, 1)
        alpha1 = np.repeat(x, 512, axis=1).reshape((512, 512, 1))

        alpha2 = np.repeat(x, 512, axis=1)
        alpha2 = np.flipud(alpha2).reshape((512, 512, 1))

        expected = np.concatenate((gray, alpha1, alpha2), axis=2)
        w, h, nz = expected.shape
        tw, th, rps = 256, 256, 256

        photo = lib.Photometric.MINISBLACK
        compression = lib.Compression.NONE
        pc = lib.PlanarConfig.CONTIG

        tiled = (True, False)
        modes = ('w', 'w8')
        extra_samples = (
            lib.ExtraSamples.UNSPECIFIED,
            lib.ExtraSamples.ASSOCALPHA,
        )

        g = itertools.product(tiled, modes)
        for tiled, mode in g:
            with self.subTest(tiled=tiled, mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    t['Photometric'] = photo
                    t['Compression'] = compression

                    t['ImageWidth'] = w
                    t['ImageLength'] = h
                    t['PlanarConfig'] = pc
                    t['SamplesPerPixel'] = 3
                    t['ExtraSamples'] = extra_samples

                    t['BitsPerSample'] = 8
                    if tiled:
                        t['TileLength'] = th
                        t['TileWidth'] = tw
                    else:
                        t['RowsPerStrip'] = rps

                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(t['Photometric'], photo)
                    self.assertEqual(t['PlanarConfig'], pc)
                    self.assertEqual(t['ImageWidth'], w)
                    self.assertEqual(t['ImageLength'], h)
                    self.assertEqual(t['BitsPerSample'], (8, 8, 8))
                    self.assertEqual(t['SamplesPerPixel'], 3)
                    if tiled:
                        self.assertEqual(t['TileWidth'], tw)
                        self.assertEqual(t['TileLength'], th)
                    else:
                        self.assertEqual(t['RowsPerStrip'], rps)
                    self.assertEqual(t['Compression'], compression)

                    self.assertEqual(t['ExtraSamples'], extra_samples)

                    actual = t[:]

                np.testing.assert_equal(actual, expected)

    def test_write_read_alpha(self):
        """
        Scenario: Write the scikit-image "astronaut" with an alpha layer.

        Expected Result:  The image should be lossless.
        """
        # Create a gradient alpha layer.
        x = np.arange(0, 256, 0.5).astype(np.uint8).reshape(512, 1)
        alpha = np.repeat(x, 512, axis=1).reshape((512, 512, 1))

        expected = np.concatenate((skimage.data.astronaut(), alpha), axis=2)
        w, h, nz = expected.shape
        tw, th, rps = 256, 256, 256

        photo = lib.Photometric.RGB
        compression = lib.Compression.NONE
        pc = lib.PlanarConfig.CONTIG

        tiled = (True, False)
        modes = ('w', 'w8')
        extra_samples = (
            lib.ExtraSamples.UNSPECIFIED,
            lib.ExtraSamples.ASSOCALPHA,
            lib.ExtraSamples.UNASSALPHA
        )

        g = itertools.product(tiled, modes, extra_samples)
        for tiled, mode, alpha in g:
            with self.subTest(tiled=tiled, mode=mode, alpha=alpha):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    t['Photometric'] = photo
                    t['Compression'] = compression

                    t['ImageWidth'] = w
                    t['ImageLength'] = h
                    t['PlanarConfig'] = pc
                    t['ExtraSamples'] = alpha

                    t['BitsPerSample'] = 8
                    t['SamplesPerPixel'] = nz
                    if tiled:
                        t['TileLength'] = th
                        t['TileWidth'] = tw
                    else:
                        t['RowsPerStrip'] = rps

                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(t['Photometric'], photo)
                    self.assertEqual(t['PlanarConfig'], pc)
                    self.assertEqual(t['ImageWidth'], w)
                    self.assertEqual(t['ImageLength'], h)
                    self.assertEqual(t['BitsPerSample'], (8, 8, 8, 8))
                    self.assertEqual(t['SamplesPerPixel'], nz)
                    if tiled:
                        self.assertEqual(t['TileWidth'], tw)
                        self.assertEqual(t['TileLength'], th)
                    else:
                        self.assertEqual(t['RowsPerStrip'], rps)
                    self.assertEqual(t['Compression'], compression)

                    self.assertEqual(t['ExtraSamples'], alpha)

                    actual = t[:]

                np.testing.assert_equal(actual, expected)

    def test_write_read_rgb_jpeg(self):
        """
        Scenario: Write the scikit-image "astronaut" as rgb/jpeg.

        Expected Result:  The image should be lossy to some degree.
        """
        expected = skimage.data.astronaut()

        photometrics = (lib.Photometric.RGB,)
        Compressions = (lib.Compression.JPEG,)
        planars = (lib.PlanarConfig.CONTIG,)
        tiled = (True, False)
        modes = ('w', 'w8')

        g = itertools.product(
            photometrics, Compressions, planars, tiled, modes
        )
        for photometric, compression, planar_config, tiled, mode in g:
            with self.subTest(photometric=photometric,
                              compression=compression,
                              planar_config=planar_config,
                              tiled=tiled,
                              mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    t['Photometric'] = photometric

                    w, h, nz = expected.shape
                    t['ImageWidth'] = expected.shape[1]
                    t['ImageLength'] = expected.shape[0]
                    t['PlanarConfig'] = planar_config

                    t['BitsPerSample'] = 8
                    t['SamplesPerPixel'] = 3
                    if tiled:
                        tw, th = int(w / 2), int(h / 2)
                        t['TileLength'] = th
                        t['TileWidth'] = tw
                    else:
                        rps = int(expected.shape[0] / 2)
                        t['RowsPerStrip'] = rps
                    t['Compression'] = compression

                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(t['Photometric'], photometric)
                    self.assertEqual(t['PlanarConfig'], planar_config)
                    self.assertEqual(t['ImageWidth'], w)
                    self.assertEqual(t['ImageLength'], h)
                    self.assertEqual(t['BitsPerSample'], (8, 8, 8))
                    self.assertEqual(t['SamplesPerPixel'], 3)
                    if tiled:
                        self.assertEqual(t['TileWidth'], tw)
                        self.assertEqual(t['TileLength'], th)
                    else:
                        self.assertEqual(t['RowsPerStrip'], rps)
                    self.assertEqual(t['Compression'], compression)

                    actual = t[:]

                metric = skimage.measure.compare_psnr(expected, actual)
                self.assertTrue(metric > 37)

    def test_write_read_rgb(self):
        """
        Scenario: Write the scikit-image "astronaut" to file.

        Expected Result:  The image should be lossless for the appropriate
        Compression schemes.
        """
        expected = skimage.data.astronaut()

        photometrics = (lib.Photometric.RGB,)
        Compressions = (lib.Compression.NONE, lib.Compression.LZW,
                        lib.Compression.PACKBITS, lib.Compression.DEFLATE,
                        lib.Compression.ADOBE_DEFLATE, lib.Compression.LZMA)
        planars = (lib.PlanarConfig.CONTIG,)
        tiled = (True, False)
        modes = ('w', 'w8')

        g = itertools.product(
            photometrics, Compressions, planars, tiled, modes
        )
        for photometric, compression, planar_config, tiled, mode in g:
            with self.subTest(photometric=photometric,
                              compression=compression,
                              planar_config=planar_config,
                              tiled=tiled,
                              mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    t['Photometric'] = photometric

                    w, h, nz = expected.shape
                    t['ImageWidth'] = expected.shape[1]
                    t['ImageLength'] = expected.shape[0]
                    t['PlanarConfig'] = planar_config

                    t['BitsPerSample'] = 8
                    t['SamplesPerPixel'] = 3
                    if tiled:
                        tw, th = int(w / 2), int(h / 2)
                        t['TileLength'] = th
                        t['TileWidth'] = tw
                    else:
                        rps = int(expected.shape[0] / 2)
                        t['RowsPerStrip'] = rps
                    t['Compression'] = compression
                    t['ResolutionUnit'] = lib.ResolutionUnit.INCH
                    t['XResolution'] = 7.5
                    t['YResolution'] = 7.5

                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(t['Photometric'], photometric)
                    self.assertEqual(t['PlanarConfig'], planar_config)
                    self.assertEqual(t['ImageWidth'], w)
                    self.assertEqual(t['ImageLength'], h)
                    self.assertEqual(t['BitsPerSample'], (8, 8, 8))
                    self.assertEqual(t['SamplesPerPixel'], 3)
                    if tiled:
                        self.assertEqual(t['TileWidth'], tw)
                        self.assertEqual(t['TileLength'], th)
                    else:
                        self.assertEqual(t['RowsPerStrip'], rps)
                    self.assertEqual(t['Compression'], compression)

                    self.assertEqual(t['ResolutionUnit'],
                                     lib.ResolutionUnit.INCH)
                    self.assertEqual(t['XResolution'], 7.5)
                    self.assertEqual(t['YResolution'], 7.5)

                    actual = t[:]

                np.testing.assert_equal(actual, expected)

    def test_write_read_greyscale(self):
        """
        Scenario: Write the scikit-image "camera" to file.

        Expected Result:  The data should be round-trip the same for greyscale
        photometric interpretations and lossless Compression schemes.
        """
        expected = skimage.data.camera()

        photometrics = (lib.Photometric.MINISBLACK, lib.Photometric.MINISWHITE)
        compressions = (lib.Compression.NONE, lib.Compression.LZW,
                        lib.Compression.PACKBITS, lib.Compression.DEFLATE,
                        lib.Compression.ADOBE_DEFLATE, lib.Compression.LZMA)
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
                    t['Photometric'] = photometric

                    w, h = expected.shape
                    t['ImageWidth'] = expected.shape[1]
                    t['ImageLength'] = expected.shape[0]

                    t['BitsPerSample'] = 8
                    t['SamplesPerPixel'] = 1
                    if tiled:
                        tw, th = int(w / 2), int(h / 2)
                        t['TileLength'] = th
                        t['TileWidth'] = tw
                    else:
                        rps = int(expected.shape[0] / 2)
                        t['RowsPerStrip'] = rps
                    t['Compression'] = compression

                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(t['Photometric'], photometric)
                    self.assertEqual(t['ImageWidth'], w)
                    self.assertEqual(t['ImageLength'], h)
                    self.assertEqual(t['BitsPerSample'], 8)
                    self.assertEqual(t['SamplesPerPixel'], 1)
                    if tiled:
                        self.assertEqual(t['TileWidth'], tw)
                        self.assertEqual(t['TileLength'], th)
                    else:
                        self.assertEqual(t['RowsPerStrip'], rps)
                    self.assertEqual(t['Compression'], compression)

                    actual = t[:]

                np.testing.assert_equal(actual, expected)

    def test_write_read_unequally_partitioned_grayscale_images(self):
        """
        Scenario: Write the scikit-image "camera" to file.  The tiles and
        strips do not equally partition the image.

        Expected Result:  The data should be round-trip the same for greyscale
        photometric interpretations and lossless Compression schemes.
        """
        expected = skimage.data.camera()

        for tiled in True, False:
            with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                t = TIFF(tfile.name, mode='w')
                t['Photometric'] = lib.Photometric.MINISBLACK

                w, h = expected.shape
                t['ImageWidth'] = expected.shape[1]
                t['ImageLength'] = expected.shape[0]

                t['BitsPerSample'] = 8
                t['SamplesPerPixel'] = 1
                if tiled:
                    tw, th = 160, 160
                    t['TileLength'] = th
                    t['TileWidth'] = tw
                else:
                    rps = 160
                    t['RowsPerStrip'] = rps
                t['Compression'] = lib.Compression.NONE

                t[:] = expected

                del t

                t = TIFF(tfile.name)
                self.assertEqual(t['Photometric'], lib.Photometric.MINISBLACK)
                self.assertEqual(t['ImageWidth'], w)
                self.assertEqual(t['ImageLength'], h)
                self.assertEqual(t['BitsPerSample'], 8)
                self.assertEqual(t['SamplesPerPixel'], 1)
                if tiled:
                    self.assertEqual(t['TileWidth'], tw)
                    self.assertEqual(t['TileLength'], th)
                else:
                    self.assertEqual(t['RowsPerStrip'], rps)
                self.assertEqual(t['Compression'], lib.Compression.NONE)

                actual = t[:]

            np.testing.assert_equal(actual, expected)

    def test_write_read_unequally_partitioned_rgb_images(self):
        """
        Scenario: Write the scikit-image RGB "astronaut" image to file.
        The tiles and strips do not equally partition the image.

        Expected Result:  The data should be round-trip the same for RGB
        photometric interpretations and lossless Compression schemes.
        """
        expected = skimage.data.astronaut()

        for tiled in True, False:
            with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                t = TIFF(tfile.name, mode='w')
                t['Photometric'] = lib.Photometric.RGB

                w, h, spp = expected.shape
                t['ImageWidth'] = w
                t['ImageLength'] = h
                t['SamplesPerPixel'] = spp
                t['PlanarConfig'] = lib.PlanarConfig.CONTIG

                t['BitsPerSample'] = 8
                if tiled:
                    tw, th = 160, 160
                    t['TileLength'] = th
                    t['TileWidth'] = tw
                else:
                    rps = 160
                    t['RowsPerStrip'] = rps
                t['Compression'] = lib.Compression.NONE

                t[:] = expected

                del t

                t = TIFF(tfile.name)
                self.assertEqual(t['Photometric'], lib.Photometric.RGB)
                self.assertEqual(t['ImageWidth'], w)
                self.assertEqual(t['ImageLength'], h)
                self.assertEqual(t['BitsPerSample'], (8, 8, 8))
                self.assertEqual(t['SamplesPerPixel'], 3)
                if tiled:
                    self.assertEqual(t['TileWidth'], tw)
                    self.assertEqual(t['TileLength'], th)
                else:
                    self.assertEqual(t['RowsPerStrip'], rps)
                self.assertEqual(t['Compression'], lib.Compression.NONE)

                actual = t[:]

            np.testing.assert_equal(actual, expected)

    def test_write_read_char_tags_datetime_tag(self):
        """
        Scenario: Write an image with char tags and the Datetime tag.  The
        datetime tag value should be provided as a datetime.datetime value.

        Expected Result:  The char tags should match roundtrip.
        """
        expected = skimage.data.astronaut()

        with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
            t = TIFF(tfile.name, mode='w')
            t['Photometric'] = lib.Photometric.RGB
            t['Compression'] = lib.Compression.NONE

            w, h, spp = expected.shape
            t['ImageWidth'] = w
            t['ImageLength'] = h
            t['SamplesPerPixel'] = spp
            t['PlanarConfig'] = lib.PlanarConfig.CONTIG

            t['BitsPerSample'] = 8

            tw, th = 160, 160
            t['TileLength'] = th
            t['TileWidth'] = tw

            t0 = datetime.datetime(2018, 5, 28, 12, 0)
            t['Datetime'] = t0

            software = 'Spacely Sprockets'
            t['Software'] = software

            t[:] = expected

            del t

            t = TIFF(tfile.name)

            self.assertEqual(t['Software'], software)
            self.assertEqual(t['Datetime'], t0)

            actual = t[:]

        np.testing.assert_equal(actual, expected)
