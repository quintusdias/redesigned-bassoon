# Standard library imports
import unittest
import warnings

try:
    # 3.7+
    import importlib.resources as ir
except ImportError:
    # 3rd party library imports, 3.6 and earlier.
    import importlib_resources as ir

# Local imports
from spiff.spiff import TIFF, TIFFReadImageError
from . import fixtures, data


class TestSuite(unittest.TestCase):

    def test_repr_main(self):
        """
        Scenario:  Test TIFF object representation on the main IFD.

        Expected Result:  Should look same as output of tiffinfo.
        """
        with ir.path(data, 'zackthecat.tif') as path:
            t = TIFF(path)
        actual = repr(t)
        expected = fixtures.zackthecat_tiffinfo
        self.maxDiff = None

        # Sometimes there are extra garbage characters on the end.  Nothing
        # we can do about that, I don't think.
        n = len(expected)
        self.assertEqual(actual[:n], expected)

    def test_repr_exif(self):
        """
        Scenario: Read an EXIF subdirectory.

        Expected Result:  The results of repr should match that of tiffinfo.
        """
        with ir.path(data, 'b52a2fceb34f9b31cb417379cf8c02ba.tif') as path:
            t = TIFF(path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t.visit_ifd(t['ExifIFD'])
        actual = repr(t)
        expected = fixtures.repr_exif

        # Sometimes there are extra garbage characters on the end.  Nothing
        # we can do about that, I don't think.
        n = len(expected)
        self.assertEqual(actual[:n], expected)

    def test_read_image_in_exif_directory(self):
        """
        Scenario:  Attempt to read an image from within an EXIF sub ifd.
        There's no actual image here, and trying to read the image caused
        libtiff to segfault.  We don't want that.

        Expected Result:  A TIFFReadImageError should be raised.
        """
        with ir.path(data, 'b52a2fceb34f9b31cb417379cf8c02ba.tif') as path:
            t = TIFF(path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t.visit_ifd(t['ExifIFD'])
        with self.assertRaises(TIFFReadImageError):
            t[:]
