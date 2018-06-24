------------
How do I...?
------------


... create a TIFF with subIFDs?
===============================
The workflow here is very similar to LibTIFF's C API.  Here, you need only
supply the number of IFDs you will be writing, then use next_image when you are
finished with the primary IFD.

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
