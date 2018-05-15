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
