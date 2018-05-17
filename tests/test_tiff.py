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
from tiffany.lib import Compression, Photometric


class TestSuite(unittest.TestCase):

    def test_write_camera(self):
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

        for photometric, compression, tiled in itertools.product(photometrics,
                                                                 compressions,
                                                                 tiled):
            with self.subTest(photometric=photometric,
                              compression=compression,
                              tiled=tiled):
                with tempfile.NamedTemporaryFile(suffix='.tif') as tfile:
                    t = TIFF(tfile.name, mode='w')
                    t['photometric'] = photometric
                    t['imagewidth'] = expected.shape[1]
                    t['imagelength'] = expected.shape[0]
                    t['bitspersample'] = 8
                    t['samplesperpixel'] = 1
                    if tiled:
                        t['tilewidth'] = int(expected.shape[1] / 2)
                        t['tilelength'] = int(expected.shape[0] / 2)
                    else:
                        t['rowsperstrip'] = int(expected.shape[0] / 2)
                    t['compression'] = compression
                    t[:] = expected

                actual = t[:]
                np.testing.assert_equal(actual, expected)
