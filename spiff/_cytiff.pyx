#cython: language=c++

from libcpp.string cimport string
from libc.stdio cimport FILE, fopen, fread, fclose, tmpfile, rewind, ftell

cimport tiff

cpdef print_directory(unsigned long tiffp):
    """
    Interfaces to the TIFF library routine TIFFPrintDirectory.

    Couldn't seem to figure out a way to do this through ctypes.

    Parameters
    ----------
    tiffp : pointer
        "file pointer" returned by TIFFOpen routine.
    """
    cdef FILE *fp
    cdef char *c_string
    cdef long numbytes
    cdef bytes py_string
    cdef string s

    # Print the directory description into a temporary file.
    fp = tmpfile()
    tiff.TIFFPrintDirectory(<tiff.TIFF *>tiffp, fp, 0)

    # How many bytes were printed?  Need to make our returned string at least
    # this big.
    numbytes = ftell(fp)
    s.reserve(numbytes)

    # Rewind and reread the contents.
    rewind(fp)
    fread(&s[0], 1, numbytes, fp)
    fclose(fp)

    # Convert to python bytes and return to Python.
    c_string = <char *>s.c_str()
    py_string = c_string
    return py_string
