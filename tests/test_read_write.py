# Local imports
import datetime
import itertools
import pathlib
import platform
import tempfile
import unittest
import warnings

# Third party library imports
import numpy as np
import skimage.data
import skimage.measure

# Local imports
from spiff.spiff import TIFF, JPEGColorModeRawError, DatatypeMismatchError
from spiff.lib import LibTIFFError
from spiff import lib


@unittest.skipIf(platform.system() == 'Windows', "tempfile issue on Windows")
class TestSuite(unittest.TestCase):

    def _get_path(self, filename):
        """
        Return full path of a test file.
        """
        directory = pathlib.Path(__file__).parent
        return directory / 'data' / filename

    def _verify_lzw(self, tfile, tiled, mode, predictor, expected):
        photo = lib.Photometric.RGB
        comp = lib.Compression.LZW
        pc = lib.PlanarConfig.CONTIG

        t = TIFF(tfile.name, mode=mode)
        t['Photometric'] = photo
        t['Compression'] = comp
        t['Predictor'] = predictor
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
        self.assertEqual(t['Predictor'], predictor)

        actual = t[:]

        np.testing.assert_array_equal(actual, expected)

    def test_write_read_lzw_predictor_integer(self):
        """
        Scenario: Write the scikit-image "astronaut" with lzw compression and
        integer predictor.

        Expected Result:  Integer predictor compression is superior to no
        compression.
        """
        expected = skimage.data.astronaut()

        predictors = (lib.Predictor.NONE, lib.Predictor.HORIZONTAL)

        tiled = (True, False)
        modes = ('w', 'w8')

        g = itertools.product(tiled, modes, predictors)
        for tiled, mode, predictor in g:
            with self.subTest(tiled=tiled, mode=mode, predictor=predictor):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    self._verify_lzw(tfile, tiled, mode, predictor, expected)

                    p = pathlib.Path(tfile.name)
                    sz = p.stat().st_size

                    if predictor == lib.Predictor.NONE:
                        self.assertTrue(sz > 740000)
                    else:
                        self.assertTrue(sz < 540000)

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

    def test_error(self):
        """
        Scenario: Write to an image that has been opened read-only.

        Expected Result:  A LibTIFFError exception is raised.
        """
        expected = skimage.data.astronaut()
        with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
            t = TIFF(tfile.name, 'w')
            t['Photometric'] = lib.Photometric.RGB

            w, h, nz = expected.shape
            t['ImageWidth'] = w
            t['ImageLength'] = h
            t['PlanarConfig'] = lib.PlanarConfig.CONTIG
            t['Compression'] = lib.Compression.NONE
            t['BitsPerSample'] = 8
            t['SamplesPerPixel'] = 3

            rps = int(h / 2)
            t['RowsPerStrip'] = rps

            t[:] = expected

            del t

            t = TIFF(tfile.name, mode='r')
            with self.assertRaises(LibTIFFError):
                t[:] = expected

    def test_ycbcr_jpeg_with_bad_datatype(self):
        """
        Scenario: Write floating point data as YCbCr/JPEG.

        Expected Result:  RuntimeError
        """
        expected = skimage.data.coffee().astype(np.float64)

        with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
            t = TIFF(tfile.name, mode='w')
            t['Photometric'] = lib.Photometric.YCBCR
            t['Compression'] = lib.Compression.JPEG
            t['JPEGColorMode'] = lib.JPEGColorMode.RGB
            t['JPEGQuality'] = 75
            t['YCbCrSubsampling'] = (1, 1)

            w, h, nz = expected.shape
            t['ImageWidth'] = expected.shape[1]
            t['ImageLength'] = expected.shape[0]
            t['PlanarConfig'] = lib.PlanarConfig.CONTIG
            t['BitsPerSample'] = 8
            t['SamplesPerPixel'] = 3

            t['TileLength'] = 64
            t['TileWidth'] = 64

            with self.assertRaises(DatatypeMismatchError):
                t[:] = expected

    def test_write_read_ycbcr_jpeg_rgb(self):
        """
        Scenario: Write the scikit-image "astronaut" as ycbcr/jpeg.

        Expected Result:  The image should be lossy to some degree.
        """
        expected = skimage.data.astronaut()

        photometric = lib.Photometric.YCBCR
        compression = lib.Compression.JPEG
        pc = lib.PlanarConfig.CONTIG
        # subsamplings = ((1, 1), (1, 2), (2, 1), (2, 2))
        qualities = (50, 75, 100)
        tiled = (True, False)
        modes = ('w', 'w8')

        g = itertools.product(tiled, qualities, modes)
        for tiled, quality, mode in g:
            with self.subTest(tiled=tiled,
                              quality=quality,
                              mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    t['Photometric'] = photometric
                    t['Compression'] = compression
                    t['JPEGColorMode'] = lib.JPEGColorMode.RGB
                    t['JPEGQuality'] = quality
                    t['YCbCrSubsampling'] = (1, 1)

                    w, h, nz = expected.shape
                    t['ImageWidth'] = expected.shape[1]
                    t['ImageLength'] = expected.shape[0]
                    t['PlanarConfig'] = pc
                    t['BitsPerSample'] = 8
                    t['SamplesPerPixel'] = 3

                    if tiled:
                        tw, th = int(w / 2), int(h / 2)
                        t['TileLength'] = th
                        t['TileWidth'] = tw
                    else:
                        rps = int(expected.shape[0] / 2)
                        t['RowsPerStrip'] = rps

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
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

                    t.rgba = True

                    # Get rid of the alpha layer.
                    actual = t[:][:, :, :3]

                metric = skimage.measure.compare_psnr(expected, actual)
                if quality == 50:
                    self.assertTrue(metric > 33)
                elif quality == 75:
                    self.assertTrue(metric > 35)
                elif quality == 100:
                    self.assertTrue(metric > 50)

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

    def _set_tags(self, t, tags, tiled):
        """
        Set all the TIFF tags.
        """

        if tiled:
            tags['TileLength'] = int(tags['ImageLength'] / 2)
            tags['TileWidth'] = int(tags['ImageWidth'] / 2)
        else:
            tags['RowsPerStrip'] = int(tags['ImageLength'] / 2)

        for tag, value in tags.items():
            t[tag] = value

    def test_write_read_subifds(self):
        """
        Scenario: Write an IFD for the scikit-image astronaut, including two
        subIFDs of the same image.

        Expected Result:  The length of the tiff object should only be one,
        but the SubIFD tag should indicate two subIFDs.
        """
        expected = skimage.data.astronaut()
        w, h, nz = expected.shape

        tiled = (True, False)
        modes = ('w', 'w8')

        tags = {
            'Photometric': lib.Photometric.RGB,
            'ImageWidth': w,
            'ImageLength': h,
            'PlanarConfig': lib.PlanarConfig.CONTIG,
            'BitsPerSample': 8,
            'SamplesPerPixel': 3,
            'Compression': lib.Compression.NONE,
        }

        for tiled, mode in itertools.product(tiled, modes):
            with self.subTest(tiled=tiled, mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    self._set_tags(t, tags, tiled)

                    # Now write the sub IFD tag.
                    t['SubIFDs'] = 2

                    # And finally write the primary image.
                    t[:] = expected

                    # Position to the first subIFD
                    t.new_image()
                    self._set_tags(t, tags, tiled)
                    t['ImageDescription'] = 'SubIFD #1'
                    t[:] = expected

                    # Position to the second subIFD
                    t.new_image()
                    self._set_tags(t, tags, tiled)
                    t['ImageDescription'] = 'SubIFD #2'
                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)

                    # Verify the primary image.
                    actual = t[:]
                    np.testing.assert_equal(actual, expected)

                    # Verify there are no following IFDs, that the two other
                    # images were written as subIFDs.
                    self.assertEqual(len(t), 1)
                    self.assertEqual(len(t['SubIFDs']), 2)

                    # Verify the first subIFD.
                    t.visit_ifd(t['SubIFDs'][0])
                    actual = t[:]
                    np.testing.assert_equal(actual, expected)
                    self.assertEqual(t['ImageDescription'], 'SubIFD #1')

                    t.back()

                    # Verify the second subIFD.
                    t.visit_ifd(t['SubIFDs'][1])
                    actual = t[:]
                    np.testing.assert_equal(actual, expected)
                    self.assertEqual(t['ImageDescription'], 'SubIFD #2')

    def test_write_read_ifds(self):
        """
        Scenario: Write three copies of the the scikit-image "astronaut" to
        file in three separate IFDs.  Then read them back.

        Expected Result:  The iteration protocol should yield all three.
        """
        expected = skimage.data.astronaut()
        w, h, nz = expected.shape

        tiled = (True, False)
        modes = ('w', 'w8')

        tags = {
            'Photometric': lib.Photometric.RGB,
            'ImageWidth': w,
            'ImageLength': w,
            'PlanarConfig': lib.PlanarConfig.CONTIG,
            'BitsPerSample': 8,
            'SamplesPerPixel': 3,
            'Compression': lib.Compression.NONE,
        }

        for tiled, mode in itertools.product(tiled, modes):
            with self.subTest(tiled=tiled, mode=mode):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode=mode)
                    self._set_tags(t, tags, tiled)
                    t[:] = expected

                    t.new_image()
                    self._set_tags(t, tags, tiled)
                    t[:] = expected

                    t.new_image()
                    self._set_tags(t, tags, tiled)
                    t[:] = expected

                    del t

                    t = TIFF(tfile.name)
                    self.assertEqual(len(t), 3)

                    actuals = [t[:] for t in TIFF(tfile.name)]
                    for actual in actuals:
                        np.testing.assert_equal(actual, expected)

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
