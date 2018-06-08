# Local imports
import unittest

# Local imports
from spiff import lib


class TestSuite(unittest.TestCase):

    def test_version(self):
        """
        Scenario: Call the TIFFGetVersion library routine.

        Expected Result:  Get an appropriate version string back.
        """
        version = lib.getVersion()
        self.assertRegex(version, 'LIBTIFF, Version \d.\d+.\d+')
