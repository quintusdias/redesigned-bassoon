# Standard library imports ...
import doctest

# Local imports
import spiff

def load_tests(loader, tests, ignore):
    """
    Run doc tests
    """
    tests.addTests(doctest.DocTestSuite('spiff.lib'))
    return tests

