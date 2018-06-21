# Local imports
import pathlib
import unittest
import warnings

# Third party library imports
import numpy as np

# Local imports
from spiff.spiff import TIFF, TIFFReadImageError, NoEXIFIFDError
from spiff.lib import (
    Compression, Photometric, PlanarConfig, JPEGProc,
    ResolutionUnit, SampleFormat, NotRGBACompatibleError
)
from . import fixtures


class TestSuite(unittest.TestCase):

    def _get_path(self, filename):
        """
        Return full path of a test file.
        """
        directory = pathlib.Path(__file__).parent
        return directory / 'data' / filename

    def test_repr_main(self):
        """
        Scenario:  Test TIFF object representation on the main IFD. 

        Expected Result:  Should look same as output of tiffinfo.
        """
        path = self._get_path('zackthecat.tif')
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
        path = self._get_path('b52a2fceb34f9b31cb417379cf8c02ba.tif')
        t = TIFF(path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t.visit_ifd(t['ExifIFD'])
        actual = repr(t)
        expected = fixtures.repr_exif
        self.assertEqual(actual, expected)

    def test_read_image_in_exif_directory(self):
        """
        Scenario:  Attempt to read an image from within an EXIF sub ifd.
        There's no actual image here, and trying to read the image caused
        libtiff to segfault.  We don't want that.

        Expected Result:  A TIFFReadImageError should be raised.
        """
        path = self._get_path('b52a2fceb34f9b31cb417379cf8c02ba.tif')
        t = TIFF(path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t.visit_ifd(t['ExifIFD'])
        with self.assertRaises(TIFFReadImageError):
            t[:]
