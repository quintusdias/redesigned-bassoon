from libc.stdio cimport FILE, fopen, fread, printf, fscanf, tmpfile, rewind, ftell

cimport tiff

cpdef print_directory(unsigned long tiffp):
    cdef FILE *fp
    cdef long numbytes
    cdef bytes py_string

    cdef char str[1024]

    fp = tmpfile()
    tiff.TIFFPrintDirectory(<tiff.TIFF *>tiffp, fp, 0)

    numbytes = ftell(fp)

    # Rewind and reread the contents.
    rewind(fp)
    fread(str, 1, numbytes, fp)
    py_string = str
    return py_string
