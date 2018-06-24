# Standard library imports ...
import doctest


def load_tests(loader, tests, ignore):
    """
    Run doc tests
    """
    tests.addTests(doctest.DocTestSuite('spiff.lib'))
    return tests
