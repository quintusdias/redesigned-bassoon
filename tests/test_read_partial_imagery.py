# Local imports
import pathlib
import unittest

# Third party library imports
import numpy as np

# Local imports
from spiff.spiff import TIFF


class TestSuite(unittest.TestCase):

    def _get_path(self, filename):
        """
        Return full path of a test file.
        """
        directory = pathlib.Path(__file__).parent
        return directory / 'data' / filename

    def test_read_partial_stripped(self):
        """
        Scenario:  Read a stripped TIFF (rps=3) with contiguous planar
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
