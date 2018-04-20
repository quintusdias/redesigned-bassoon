"""
Configure glymur to use installed libraries if possible.
"""
import ctypes
from ctypes.util import find_library
import os
import platform
import sys
import warnings

if sys.hexversion <= 0x03000000:
    from ConfigParser import SafeConfigParser as ConfigParser
    from ConfigParser import NoOptionError, NoSectionError
else:
    from configparser import ConfigParser
    from configparser import NoOptionError, NoSectionError

# default library locations for MacPorts
_macports_default_location = {'openjp2': '/opt/local/lib/libopenjp2.dylib',
                              'openjpeg': '/opt/local/lib/libopenjpeg.dylib'}


def load_tiff_library(libname):

    if ((('Anaconda' in sys.version) or
         ('Continuum Analytics, Inc.' in sys.version) or
         ('packaged by conda-forge' in sys.version))):
        # If Anaconda, then openjpeg may have been installed via conda.
        if platform.system() in ['Linux', 'Darwin']:
            suffix = '.so' if platform.system() == 'Linux' else '.dylib'
            basedir = os.path.dirname(os.path.dirname(sys.executable))
            lib = os.path.join(basedir, 'lib', 'lib' + libname + suffix)
        elif platform.system() == 'Windows':
            basedir = os.path.dirname(sys.executable)
            lib = os.path.join(basedir, 'Library', 'bin', libname + '.dll')

        if os.path.exists(lib):
            path = lib

    if path is None:
        # Can ctypes find it in the default system locations?
        path = find_library(libname)

    if path is None:
        if platform.system() == 'Darwin':
            # OpenJPEG may have been installed via MacPorts
            path = _macports_default_location[libname]

        if path is not None and not os.path.exists(path):
            # the mac/win default location does not exist.
            return None

    return load_library_handle(libname, path)


def load_library_handle(libname, path):
    """Load the library, return the ctypes handle."""

    if path is None or path in ['None', 'none']:
        # Either could not find a library via ctypes or
        # user-configuration-file, or we could not find it in any of the
        # default locations, or possibly the user intentionally does not want
        # one of the libraries to load.
        return None

    try:
        if os.name == "nt":
            opj_lib = ctypes.windll.LoadLibrary(path)
        else:
            opj_lib = ctypes.CDLL(path)
    except (TypeError, OSError):
        msg = 'The {libname} library at {path} could not be loaded.'
        msg = msg.format(path=path, libname=libname)
        warnings.warn(msg, UserWarning)
        opj_lib = None

    return opj_lib


def load_library():
    """
    Try to ascertain locations of the tiff library.

    Returns
    -------
    library handle
    """
    handle = load_tiff_library('tiff')

    if handle is None:
        msg = "The tiff library could be loaded.  "
        warnings.warn(msg)
    return handle
