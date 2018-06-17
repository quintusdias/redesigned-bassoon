#cython: language=c++

from libcpp.string cimport string
from libc.stdio cimport FILE, fopen, fread, printf, fscanf, tmpfile, rewind, ftell

cimport tiff

cpdef print_directory(unsigned long tiffp):
    cdef FILE *fp
    cdef char *buffer
    cdef long numbytes
    cdef bytes py_string
    cdef string s

    # cdef char str[1024]

    fp = tmpfile()
    tiff.TIFFPrintDirectory(<tiff.TIFF *>tiffp, fp, 0)

    numbytes = ftell(fp)
    s.reserve(numbytes)

    # Rewind and reread the contents.
    rewind(fp)

    fread(&s[0], 1, numbytes, fp)
    cdef char *c_string = <char *>s.c_str()
    py_string = c_string
    return py_string
