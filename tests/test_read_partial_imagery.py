# Standard library imports
import pathlib
import unittest

try:
    # 3.7+
    import importlib.resources as ir
except ImportError:
    # 3rd party library imports, 3.6 and earlier.
    import importlib_resources as ir

# Third party library imports
import numpy as np

# Local imports
import spiff
from spiff.spiff import TIFF
from . import data


class TestSuite(unittest.TestCase):

    def test_read_partial_tiled_separate_contained_in_one_tile(self):
        """
        Scenario:  Read a tiled TIFF (l=w=16) with separate planar
        configuration.  The read operation should be contained in a single
        tile.  In this case, it is tile #5.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-tiled16-separate-08.tif') as path:
            t = TIFF(path)
        actual = t[30:32, 29:31, :]

        t.rgba = True
        expected = t[:][30:32, 29:31, 0:3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_separate_partial_tiled_crosses_multiple_tiles(self):
        """
        Scenario:  Read a stripped TIFF (l=w=16) with separate planar
        configuration.  The read operation crosses tiles in both the horizontal
        and vertical direction.  In this case, it is tiles 1-4, 1-3.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-tiled16-separate-08.tif') as path:
            t = TIFF(path)
        actual = t[23:68, 22:63, :]

        t.rgba = True
        expected = t[:][23:68, 22:63, 0:3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_separate_full_tiled_image_using_indexing(self):
        """
        Scenario:  Read a stripped TIFF (l=w=16) with separate planar
        configuration.  The indexing given is the same as if the entire image
        were to be read.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-tiled16-separate-08.tif') as path:
            t = TIFF(path)
        actual = t[:76, :73, :]

        t.rgba = True
        expected = t[:][:76, :73, :3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_tiled_separate_past_image_extents_but_within_last_tile(self):
        """
        Scenario:  Read a tiled TIFF (l=w=16) with separate planar
        configuration.  The row and column indexing both extend past the
        logical indexing of the image in the last tile, but they are still
        within the last tile's phantom area.

        Expected Result:  The image is returned without error, but rows and
        columns past the image's logical extent are not included.
        """
        with ir.path(data, 'tiger-rgb-tiled16-separate-08.tif') as path:
            t = TIFF(path)
        actual = t[:78, :75, :]

        t.rgba = True
        expected = t[:][:, :, :3]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (76, 73, 3))

    def test_read_partial_tiled_contig_contained_in_one_tile(self):
        """
        Scenario:  Read a tiled TIFF (l=w=16) with contiguous planar
        configuration.  The read operation should be contained in a single
        tile.  In this case, it is tile #5.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-tiled16-contig-08.tif') as path:
            t = TIFF(path)
        actual = t[30:32, 29:31, :]

        t.rgba = True
        expected = t[:][30:32, 29:31, 0:3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_contig_partial_tiled_crosses_multiple_tiles(self):
        """
        Scenario:  Read a stripped TIFF (l=w=16) with contiguous planar
        configuration.  The read operation crosses tiles in both the horizontal
        and vertical direction.  In this case, it is tiles 1-4, 1-3.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-tiled16-contig-08.tif') as path:
            t = TIFF(path)
        actual = t[23:68, 22:63, :]

        t.rgba = True
        expected = t[:][23:68, 22:63, 0:3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_contig_full_tiled_image_using_indexing(self):
        """
        Scenario:  Read a stripped TIFF (l=w=16) with contiguous planar
        configuration.  The indexing given is the same as if the entire image
        were to be read.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-tiled16-contig-08.tif') as path:
            t = TIFF(path)
        actual = t[:76, :73, :]

        t.rgba = True
        expected = t[:][:76, :73, :3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_tiled_contig_past_image_extents_but_within_last_tile(self):
        """
        Scenario:  Read a tiled TIFF (l=w=16) with contiguous planar
        configuration.  The row and column indexing both extend past the
        logical indexing of the image in the last tile, but they are still
        within the last tile's phantom area.

        Expected Result:  The image is returned without error, but rows and
        columns past the image's logical extent are not included.
        """
        with ir.path(data, 'tiger-rgb-tiled16-contig-08.tif') as path:
            t = TIFF(path)
        actual = t[:78, :75, :]

        t.rgba = True
        expected = t[:][:, :, :3]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (76, 73, 3))

    def test_read_partial_stripped_contig_contained_in_one_strip(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with contiguous planar
        configuration.  The read operation should be contained in a single
        strip.  In this case, it is strip #1.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-strip16-contig-08.tif') as path:
            t = TIFF(path)
        actual = t[30:32, 30:32, :]

        t.rgba = True
        expected = t[:][30:32, 30:32, 0:3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_contig_stripped_crosses_multiple_strips(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with contiguous planar
        configuration.  The read operation should be contained in a single
        strip.  In this case, it is strip #1, #2, #3, and #4.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-strip16-contig-08.tif') as path:
            t = TIFF(path)
        actual = t[23:68, 23:68, :]

        t.rgba = True
        expected = t[:][23:68, 23:68, 0:3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_contig_full_stripped_image_using_indexing(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with contiguous planar
        configuration.  The indexing given is the same as if the entire image
        were to be read.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-strip16-contig-08.tif') as path:
            t = TIFF(path)
        actual = t[:76, :73, :]

        t.rgba = True
        expected = t[:][:76, :73, :3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_stripped_contig_past_imagelength_but_within_last_strip(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with contiguous planar
        configuration.  The row indexing indicates a final row that is past the
        imagelength, but is technically within the last strip.

        Expected Result:  The image is returned without error, but rows past
        the last real row are not included.
        """
        with ir.path(data, 'tiger-rgb-strip16-contig-08.tif') as path:
            t = TIFF(path)
        actual = t[:78, :73, :]

        t.rgba = True
        expected = t[:][:, :, :3]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape[0], 76)

    def test_read_partial_stripped_separate_contained_in_one_strip(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with separate planar
        configuration.  The read operation should be contained in a single
        strip.  In this case, it is strip #1.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-strip16-separate-08.tif') as path:
            t = TIFF(path)
        actual = t[30:32, 29:32, :]

        t.rgba = True
        expected = t[:][30:32, 29:32, 0:3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_partial_stripped_separate_crosses_multiple_strips(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with separate planar
        configuration.  The read operation should be contained in a single
        strip.  In this case, it is strip #1, #2, #3, and #4.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-strip16-separate-08.tif') as path:
            t = TIFF(path)
        actual = t[23:68, 23:69, :]

        t.rgba = True
        expected = t[:][23:68, 23:69, 0:3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_full_stripped_separate_image_using_indexing(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with separate planar
        configuration.  The indexing given is the same as if the entire image
        were to be read.

        Expected Result:  The assembled image should match the result produced
        by the RGBA interface.
        """
        with ir.path(data, 'tiger-rgb-strip16-separate-08.tif') as path:
            t = TIFF(path)
        actual = t[:76, :73, :]

        t.rgba = True
        expected = t[:][:76, :73, :3]
        np.testing.assert_array_equal(actual, expected)

    def test_read_strip_separate_past_imagelength_but_within_last_strip(self):
        """
        Scenario:  Read a stripped TIFF (rps=16) with separate planar
        configuration.  The row indexing indicates a final row that is past the
        imagelength, but is technically within the last strip.

        Expected Result:  The image is returned without error, but rows past
        the last real row are not included.
        """
        with ir.path(data, 'tiger-rgb-strip16-separate-08.tif') as path:
            t = TIFF(path)
        actual = t[:78, :73, :]

        t.rgba = True
        expected = t[:][:, :, :3]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape[0], 76)
