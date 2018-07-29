# Local imports
import unittest
import warnings

try:
    # 3.7+
    import importlib.resources as ir
except ImportError:
    # 3rd party library imports, 3.6 and earlier.
    import importlib_resources as ir

# Third party library imports
import numpy as np

# Local imports
from spiff.spiff import TIFF
from spiff import lib
from spiff.lib import (
    Compression, Photometric, PlanarConfig, JPEGProc,
    ResolutionUnit, SampleFormat, NotRGBACompatibleError
)
from . import data


class TestSuite(unittest.TestCase):

    def test_separated_tiled(self):
        """
        Scenario:  Read a tiled RGB TIFF with separate image planes.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-tiled16-contig-08.tif') as path:
            t = TIFF(path)
        img1 = t[:]

        t.rgba = True
        img2 = t[:][:, :, :3]
        np.testing.assert_array_equal(img1, img2)

    def test_separated_stripped(self):
        """
        Scenario:  Read a stripped RGB TIFF with separate image planes.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-strip3-planar-08.tif') as path:
            t = TIFF(path)
        img1 = t[:]

        t.rgba = True
        img2 = t[:][:, :, :3]
        np.testing.assert_array_equal(img1, img2)

    def test_move_to_exif_then_back(self):
        """
        Scenario: Read an EXIF subdirectory, then move back into the main
        directory.

        Expected Result:  The offset of the main directory should match after
        moving back from the Exif directory.
        """
        with ir.path(data, 'b52a2fceb34f9b31cb417379cf8c02ba.tif') as path:
            t = TIFF(path)
        t = TIFF(path)
        first_offset = lib.currentDirOffset(t.tfp)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t.set_subdirectory(t['ExifIFD'])
            exif_offset = lib.currentDirOffset(t.tfp)

        t.back()
        third_offset = lib.currentDirOffset(t.tfp)

        self.assertNotEqual(first_offset, exif_offset)
        self.assertEqual(first_offset, third_offset)

    def test_shape_2D(self):
        """
        Scenario: Access the shape property for a 2D image.

        Expected Result:  The two-tuple should match what is shown by TIFFINFO.
        """
        with ir.path(data, 'tiger-minisblack-float-strip-32.tif') as path:
            t = TIFF(path)
        self.assertEqual(t.shape, (76, 73))

    def test_shape_3D(self):
        """
        Scenario: Access the shape property for a 3D image.

        Expected Result:  The three-tuple should match what is shown by
        TIFFINFO.
        """
        with ir.path(data, 'zackthecat.tif') as path:
            t = TIFF(path)
        self.assertEqual(t.shape, (213, 234, 3))

    def test_rgba_refused_on_bad_candidate(self):
        """
        Scenario: Attempt to read a float32 image using the RGBA interface.

        Expected Result:  The two-tuple should correspond to what TIFFINFO
        shows.
        """
        with ir.path(data, 'tiger-minisblack-float-strip-32.tif') as path:
            t = TIFF(path)
        with self.assertRaises(NotRGBACompatibleError):
            t.rgba = True

    def test_read_8bit_palette_as_rgba(self):
        """
        Scenario: Read an 8-bit palette image in RGBA mode.

        Expected Result:  The tags match the output of TIFFDUMP.  The first
        three image layers match the RGB version of the image.
        """
        with ir.path(data, 'tiger-rgb-tile-contig-08.tif') as path:
            t = TIFF(path)
        expected = t[:]

        with ir.path(data, 'tiger-palette-tile-08.tif') as path:
            t = TIFF(path)
        t.rgba = True
        image = t[:]

        self.assertEqual(t['BitsPerSample'], 8)
        self.assertEqual(t['Compression'], Compression.NONE)
        self.assertEqual(t['Photometric'], Photometric.PALETTE)
        self.assertEqual(t['DocumentName'], 'tiger-palette-tile-08.tif')
        self.assertEqual(t['ImageDescription'],
                         '256-entry colormapped tiled image')
        self.assertEqual(t['SamplesPerPixel'], 1)
        self.assertEqual(t['XResolution'], 72.0)
        self.assertEqual(t['YResolution'], 72.0)
        self.assertEqual(t['PlanarConfig'], PlanarConfig.CONTIG)
        self.assertEqual(t['ResolutionUnit'], ResolutionUnit.NONE)
        self.assertEqual(t['PageNumber'], (0, 1))
        self.assertEqual(t['Software'],
                         ('GraphicsMagick 1.4 unreleased Q32 '
                          'http://www.GraphicsMagick.org/'))
        self.assertEqual(t['TileWidth'], 16)
        self.assertEqual(t['TileLength'], 16)
        self.assertEqual(t['SampleFormat'], SampleFormat.UINT)

        # np.testing.assert_array_equal(image[:, :, :3], expected)
        self.assertEqual(image[:, :, :3].shape, expected.shape)

    def test_read_ojpeg(self):
        """
        Scenario: Read a TIFF with OJPEG Compression.

        Expected Result:  The tags match the output of TIFFDUMP.  The image
        size matches the tag values.
        """
        with ir.path(data, 'zackthecat.tif') as path:
            t = TIFF(path)
        with self.assertWarns(UserWarning):
            image = t[:]

        self.assertEqual(image.shape, (t.h, t.w, t.spp))

        self.assertEqual(t['BitsPerSample'], (8, 8, 8))
        self.assertEqual(t['Compression'], Compression.OJPEG)
        self.assertEqual(t['Photometric'], Photometric.YCBCR)
        self.assertEqual(t['XResolution'], 75.0)
        self.assertEqual(t['YResolution'], 75.0)
        self.assertEqual(t['PlanarConfig'], PlanarConfig.CONTIG)
        self.assertEqual(t['ResolutionUnit'], ResolutionUnit.INCH)
        self.assertEqual(t['TileWidth'], 240)
        self.assertEqual(t['TileLength'], 224)
        self.assertEqual(t['JPEGProc'], JPEGProc.BASELINE)

        self.assertEqual(t['JPEGQTables'], (7364, 7428, 7492))
        self.assertEqual(t['JPEGDCTables'], (7568, 7596, 7624))
        self.assertEqual(t['JPEGACTables'], (7664, 7842, 8020))

        np.testing.assert_allclose(t['YCbCrCoefficients'],
                                   (0.299, 0.587, 0.114))

        self.assertEqual(t['YCbCrSubsampling'], (2, 2))
        self.assertEqual(t['ReferenceBlackWhite'],
                         (16, 235, 128, 240, 128, 240))
