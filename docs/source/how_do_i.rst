------------
How do I...?
------------

... just create a TIFF, man...
==============================
At its simplest, TIFFs are images with usually very specific metadata tags.
Most of the time, you can use numpy's slicing model to handle the image data
and python' dictionary model to handle the tags.  For this RGB image, here's
the minimum you'd have to do to create a tiled TIFF.

    >>> import skimage.data
    >>> image = skimage.data.astronaut()
    >>> w, h, nz = image.shape
    >>> from spiff import TIFF, lib
    >>> t = TIFF('astronaut3.tif', mode='w')
    >>> t['Photometric'] = lib.Photometric.RGB
    >>> t['Compression'] = lib.Compression.LZW
    >>> t['PlanarConfig'] = lib.PlanarConfig.CONTIG
    >>> t['BitsPerSample'] = 8
    >>> t['ImageWidth'] = w
    >>> t['ImageLength'] = h
    >>> t['SamplesPerPixel'] = 3
    >>> t['TileWidth'] = int(w/2)
    >>> t['TileLength'] = int(h/2)
    >>> t[:] = image
    >>> t
    TIFF Directory at offset 0x0 (0)
      Image Width: 512 Image Length: 512
      Tile Width: 256 Tile Length: 256
      Bits/Sample: 8
      Compression Scheme: LZW
      Photometric Interpretation: RGB color
      Samples/Pixel: 3
      Planar Configuration: single image plane
    >>> del t

Using [:] means that you don't have to bother writing each individual tile,
spiff will handle that for you.  

If you are familiar with the libtiff's command line utility tiffinfo, you may
recognize that the spiff.TIFF class has tied its __repr__ method to libtiff's
TIFFPrintDirectory function.

... create a BigTIFF...
==============================
Easy.  Just think about how libtiff does it with the TIFFOpen function.

    >>> import skimage.data
    >>> image = skimage.data.astronaut()
    >>> w, h, nz = image.shape
    >>> from spiff import TIFF, lib
    >>> t = TIFF('astronaut3.tif', mode='w8')
    >>> t['Photometric'] = lib.Photometric.RGB
    >>> t['Compression'] = lib.Compression.LZW
    >>> t['PlanarConfig'] = lib.PlanarConfig.CONTIG
    >>> t['BitsPerSample'] = 8
    >>> t['ImageWidth'] = w
    >>> t['ImageLength'] = h
    >>> t['SamplesPerPixel'] = 3
    >>> t['TileWidth'] = int(w/2)
    >>> t['TileLength'] = int(h/2)
    >>> t[:] = image
    >>> t
    TIFF Directory at offset 0x0 (0)
      Image Width: 512 Image Length: 512
      Tile Width: 256 Tile Length: 256
      Bits/Sample: 8
      Compression Scheme: LZW
      Photometric Interpretation: RGB color
      Samples/Pixel: 3
      Planar Configuration: single image plane
    >>> del t
    >>> !file astronaut3.tif
    astronaut3.tif: Big TIFF image data, little-endian

... create a TIFF with subIFDs?
===============================
In this case, it helps to be a bit familiar with the workflow for
libtiff's C API.  Here, though, you need only supply the number of
IFDs you will be writing, then use visit_ifd when you are finished
with the primary IFD.

    >>> import skimage.data
    >>> image = skimage.data.astronaut()
    >>> w, h, nz = image.shape
    >>> from spiff import TIFF, lib
    >>> t = TIFF('astronaut3.tif', mode='w')
    >>> t['Photometric'] = lib.Photometric.RGB
    >>> t['ImageWidth'] = w
    >>> t['ImageLength'] = h
    >>> t['TileWidth'] = int(w/2)
    >>> t['TileLength'] = int(h/2)
    >>> t['PlanarConfig'] = lib.PlanarConfig.CONTIG
    >>> t['BitsPerSample'] = 8
    >>> t['SamplesPerPixel'] = 3
    >>> t['Compression'] = lib.Compression.NONE

Now write the SubIFDs tag.  We will create two SubIFD images.

    >>> t['SubIFDs'] = 2

We have to finish by writing the primary IFD image, then we can move along to
the subIFDs.  Actually, we **MUST** move along to the subIFDs next.

    >>> t[:] = image
    >>> t.new_image()

We will make the first IFD different by using LZW compression.

    >>> t['Photometric'] = lib.Photometric.RGB
    >>> t['ImageWidth'] = w
    >>> t['ImageLength'] = h
    >>> t['TileWidth'] = int(w/2)
    >>> t['TileLength'] = int(h/2)
    >>> t['PlanarConfig'] = lib.PlanarConfig.CONTIG
    >>> t['BitsPerSample'] = 8
    >>> t['SamplesPerPixel'] = 3
    >>> t['Compression'] = lib.Compression.LZW
    >>> t[:] = image

And finally, position to the second subIFD and write that one using JPEG
compression and close the file.

    >>> t.new_image()
    >>> t['Photometric'] = lib.Photometric.YCBCR
    >>> t['ImageWidth'] = w
    >>> t['ImageLength'] = h
    >>> t['TileWidth'] = int(w/2)
    >>> t['TileLength'] = int(h/2)
    >>> t['PlanarConfig'] = lib.PlanarConfig.CONTIG
    >>> t['BitsPerSample'] = 8
    >>> t['SamplesPerPixel'] = 3
    >>> t['Compression'] = lib.Compression.JPEG
    >>> t['JPEGColorMode'] = lib.JPEGColorMode.RGB
    >>> t['JPEGQuality'] = 75
    >>> t['YCbCrSubsampling'] = (1, 1)
    >>> t[:] = image
    >>> del t

When we open the file, we can verify that there is only one main IFD with 
:py:meth:`len` method

    >>> t = TIFF('astronaut3.tif')
    >>> len(t)
    1

We can also see by inspection that two subIFDs have been written.

    >>> t
    TIFF Directory at offset 0xc0008 (786440)
      Image Width: 512 Image Length: 512
      Tile Width: 256 Tile Length: 256
      Bits/Sample: 8
      Compression Scheme: None
      Photometric Interpretation: RGB color
      Samples/Pixel: 3
      Planar Configuration: single image plane
      SubIFD Offsets: 1528596 1578110

We can reach each subIFD with the visit method.

    >>> t.visit_ifd(t['SubIFDs'][1])
    >>> t
    TIFF Directory at offset 0x18147e (1578110)
      Image Width: 512 Image Length: 512
      Tile Width: 256 Tile Length: 256
      Bits/Sample: 8
      Compression Scheme: JPEG
      Photometric Interpretation: YCbCr
      YCbCr Subsampling: 1, 1
      Samples/Pixel: 3
      Planar Configuration: single image plane
      Reference Black/White:
         0:     0   255
         1:   128   255
         2:   128   255
      JPEG Tables: (574 bytes)
