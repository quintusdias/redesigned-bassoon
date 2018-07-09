# Local imports
import pathlib
import unittest

# Third party library imports
import numpy as np

# Local imports
import spiff
from spiff.spiff import TIFF


class TestSuite(unittest.TestCase):

    def _get_path(self, filename):
        """
        Return full path of a test file.
        """
        directory = pathlib.Path(__file__).parent
        return directory / 'data' / filename

    def test_read_partial_stripped_contig_contained_in_one_strip(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with contiguous planar
        configuration.  The read operation should be contained in a single
        strip.  In this case, it is strip #1.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        path = self._get_path('tiger-rgb-strip16-contig-08.tif')
        t = TIFF(path)
        actual = t[30:32, 30:32, :]

        t.rgba = True
        expected = t[:][30:32, 30:32, 0:3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_given_illegal_ranges(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with contiguous planar
        configuration, except that the indices are not valid.

        Expected Result:  An exception should be raised.
        """
        path = self._get_path('tiger-rgb-strip16-contig-08.tif')
        t = TIFF(path)
        with self.assertRaises(spiff.lib.LibTIFFError):
            t[30:800, 30:32, :]

    def test_read_partial_stripped_contig_crosses_multiple_strips(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with contiguous planar
        configuration.  The read operation should be contained in a single
        strip.  In this case, it is strip #1, #2, #3, and #4.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        path = self._get_path('tiger-rgb-strip16-contig-08.tif')
        t = TIFF(path)
        actual = t[23:68, 23:68, :]

        t.rgba = True
        expected = t[:][23:68, 23:68, 0:3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_full_stripped_contig_image_using_indexing(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with contiguous planar
        configuration.  The indexing given is the same as if the entire image
        were to be read.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        path = self._get_path('tiger-rgb-strip16-contig-08.tif')
        t = TIFF(path)
        actual = t[:76, :73, :]

        t.rgba = True
        expected = t[:][:76, :73, :3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_partial_stripped_planar_contained_in_one_strip(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with contiguous planar
        configuration.  The read operation should be contained in a single
        strip.  In this case, it is strip #1.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        path = self._get_path('tiger-rgb-strip16-planar-08.tif')
        t = TIFF(path)
        actual = t[30:32, 30:32, :]

        t.rgba = True
        expected = t[:][30:32, 30:32, 0:3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_partial_stripped_planar_crosses_multiple_strips(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with contiguous planar
        configuration.  The read operation should be contained in a single
        strip.  In this case, it is strip #1, #2, #3, and #4.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        path = self._get_path('tiger-rgb-strip16-planar-08.tif')
        t = TIFF(path)
        actual = t[23:68, 23:68, :]

        t.rgba = True
        expected = t[:][23:68, 23:68, 0:3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_full_stripped_planar_image_using_indexing(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with contiguous planar
        configuration.  The indexing given is the same as if the entire image
        were to be read.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        path = self._get_path('tiger-rgb-strip16-planar-08.tif')
        t = TIFF(path)
        actual = t[:76, :73, :]

        t.rgba = True
        expected = t[:][:76, :73, :3]
        np.testing.assert_array_equal(actual, expected)
