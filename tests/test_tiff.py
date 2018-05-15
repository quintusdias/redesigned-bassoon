# Local imports
import tempfile
import unittest

# Third party library imports
import numpy as np
import pkg_resources as pkg
import skimage.data
import skimage.measure

# Local imports
import tiffany
from tiffany.tiffany import TIFF
import tiffany.lib
from tiffany.lib import (
    Compression, Photometric, FillOrder, Orientation, PlanarConfig
)


class TestSuite(unittest.TestCase):

    def setUp(self):
        self.nemo = pkg.resource_filename(__name__, 'data/nemo.tif')

    def test_nemo_tags(self):
        """
        Scenario:  Open nemo.dump.

        Expected Result:  The tags match what one would see in
        tiffinfo/tiffdump.
        """
        t = TIFF(tiffany.data.nemo)
        self.assertEqual(t['ImageWidth'], 2592)
        self.assertEqual(t['ImageLength'], 1456)
        self.assertEqual(t['BitsPerSample'], (8, 8, 8))
        self.assertEqual(t['Compression'], Compression.JPEG)
        self.assertEqual(t['Photometric'], Photometric.YCBCR)
        self.assertEqual(t['FillOrder'], FillOrder.MSB2LSB)
        self.assertEqual(t['Orientation'], Orientation.TOPLEFT)
        self.assertEqual(t['SamplesPerPixel'], 3)
        self.assertEqual(t['PlanarConfig'], PlanarConfig.CONTIG)
        self.assertEqual(t['PageNumber'], (0, 1))

        np.testing.assert_almost_equal(t['WhitePoint'], (0.3127, 0.329),
                                       decimal=4)

        np.testing.assert_almost_equal(t['PrimaryChromaticities'],
                                       (0.64, 0.33, 0.3, 0.6, 0.15, 0.06),
                                       decimal=4)

        self.assertEqual(t['TileWidth'], 256)
        self.assertEqual(t['TileLength'], 256)
        self.assertEqual(len(t['TileOffsets']), 66)
        self.assertEqual(len(t['TileByteCounts']), 66)
        self.assertEqual(len(t['JPEGTables']), 574)

        np.testing.assert_equal(t['ReferenceBlackWhite'],
                                (0.0, 255.0, 128.0, 255.0, 128.0, 255.0))

    def test_getfield(self):
        """
        Scenario:  read TIFF tags with library

        Expected Result:  match what is returned by the TIFF class.
        """
        t = TIFF(tiffany.data.nemo)

        width = tiffany.lib.getField(t.tfp, 'width')
        self.assertEqual(width, t['ImageWidth'])

        bps = tiffany.lib.getFieldDefaulted(t.tfp, 'bitspersample')
        self.assertEqual(bps, t['BitsPerSample'][0])

    def test_write_camera(self):
        """
        Scenario: Write the scikit-image "camera" to file.

        """
        expected = skimage.data.camera()

        with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
            t = TIFF(tfile.name, mode='w')
            t['photometric'] = Photometric.MINISBLACK
            t['imagewidth'] = expected.shape[1]
            t['imagelength'] = expected.shape[0]
            t['bitspersample'] = 8
            t['samplesperpixel'] = 1
            t['tilewidth'] = int(expected.shape[1] / 2)
            t['tilelength'] = int(expected.shape[0] / 2)
            t['compression'] = Compression.NONE
            t[:] = expected

        actual = t[:]
        np.testing.assert_equal(actual, expected)
