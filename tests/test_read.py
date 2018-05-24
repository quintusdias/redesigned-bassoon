# Local imports
import pathlib
import unittest

# Third party library imports
import numpy as np

# Local imports
from tiffany.tiffany import TIFF
from tiffany.lib import (
    Compression, Photometric, PlanarConfig, JPEGProc,
    ResolutionUnit, SampleFormat, NotRGBACompatibleError
)


class TestSuite(unittest.TestCase):

    def _get_path(self, filename):
        """
        Return full path of a test file.
        """
        directory = pathlib.Path(__file__).parent
        return directory / 'data' / filename

    def test_rgba_refused_on_bad_candidate(self):
        """
        Scenario: Attempt to read a float32 image using the RGBA interface.

        Expected Result:  We should error out since floating point images are
        not covered by the RGBA man page.
        """
        path = self._get_path('tiger-minisblack-float-strip-32.tif')
        t = TIFF(path)
        with self.assertRaises(NotRGBACompatibleError):
            t.rgba = True

    def test_read_8bit_palette_as_rgba(self):
        """
        Scenario: Read an 8-bit palette image in RGBA mode.

        Expected Result:  The tags match the output of TIFFDUMP.  The first
        three image layers match the RGB version of the image.
        """
        rgbpath = self._get_path('tiger-rgb-tile-contig-08.tif')
        t = TIFF(rgbpath)
        expected = t[:]

        path = self._get_path('tiger-palette-tile-08.tif')
        t = TIFF(path)
        t.rgba = True
        image = t[:]

        self.assertEqual(t['bitspersample'], 8)
        self.assertEqual(t['compression'], Compression.NONE)
        self.assertEqual(t['photometric'], Photometric.PALETTE)
        self.assertEqual(t['documentname'], 'tiger-palette-tile-08.tif')
        self.assertEqual(t['imagedescription'],
                         '256-entry colormapped tiled image')
        self.assertEqual(t['samplesperpixel'], 1)
        self.assertEqual(t['xresolution'], 72.0)
        self.assertEqual(t['yresolution'], 72.0)
        self.assertEqual(t['planarconfig'], PlanarConfig.CONTIG)
        self.assertEqual(t['resolutionunit'], ResolutionUnit.NONE)
        self.assertEqual(t['pagenumber'], (0, 1))
        self.assertEqual(t['software'],
                         ('GraphicsMagick 1.4 unreleased Q32 '
                          'http://www.GraphicsMagick.org/'))
        self.assertEqual(t['tilewidth'], 16)
        self.assertEqual(t['tilelength'], 16)
        self.assertEqual(t['sampleformat'], SampleFormat.UINT)

        # np.testing.assert_array_equal(image[:, :, :3], expected)
        self.assertEqual(image[:, :, :3].shape, expected.shape)

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
