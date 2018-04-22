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
