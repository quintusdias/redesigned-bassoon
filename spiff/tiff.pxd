from libc.stdio cimport FILE

cdef extern from "tiffio.h":
    ctypedef struct TIFF:
        pass
    void TIFFPrintDirectory(TIFF *tif, FILE *fd, long flags)
