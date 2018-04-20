import pathlib
import unittest

import numpy as np

import glymur
import pkg_resources as pkg
import skimage.measure

import tiffany
from tiffany.tiffany import TIFF

class TestSuite(unittest.TestCase):

    def setUp(self):
        self.nemo = pkg.resource_filename(__name__, 'data/nemo.tif')

    def test_read_rgba(self):
        """
        SCENARIO:  read a full image in RGBA mode  

        EXPECTED RESULT:  The image dimensions are the same as that produced by
        Glymur.  The image is very clean 
        """
        t = TIFF(tiffany.data.nemo)
        im = t[:]

        j = glymur.Jp2k(glymur.data.nemo())
        jim = j[:]

        m = skimage.measure.compare_psnr(jim, np.flipud(im[:, :, :3]))
        self.assertTrue(m > 35)


