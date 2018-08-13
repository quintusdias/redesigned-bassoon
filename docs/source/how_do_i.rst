------------
How do I...?
------------

... just create a TIFF, man...
==============================
At its simplest, TIFFs are images with usually very specific metadata tags.
Most of the time, you can use numpy's slicing model to handle the image data
and python's dictionary model to handle the tags.  For this RGB image, here's
the minimum you'd have to do to create a tiled TIFF.

.. doctest::

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

.. testcleanup::

    >>> import pathlib
    >>> p = pathlib.Path('astronaut3.tif')
    >>> p.unlink()

Using [:] means that you don't have to bother writing each individual tile,
spiff will handle that for you.  

If you are familiar with the libtiff's command line utility ``tiffinfo``, you
may recognize that the spiff.TIFF class has tied its :py:meth:`__repr__` method
to libtiff's ``TIFFPrintDirectory`` function.

... create a multi-page TIFF ...
================================

    >>> import skimage.data                                                        
    >>> image = skimage.data.astronaut()                                           
    >>> w, h, nz = image.shape                                                     
    >>> from spiff import TIFF, lib                                                
    >>> t = TIFF('astronaut3.tif', mode='w8')                                       
    >>> # All three images will have the same tags, so set them all
    >>> # up in advance.    
    >>> tags = {
    ...     'Photometric': lib.Photometric.RGB,
    ...     'ImageWidth': w,
    ...     'ImageLength': h,
    ...     'TileWidth': int(w / 2),
    ...     'TileLength': int(h / 2),
    ...     'PlanarConfig': lib.PlanarConfig.CONTIG,
    ...     'BitsPerSample': 8,
    ...     'SamplesPerPixel': 3,
    ...     'Compression': lib.Compression.NONE,
    ... }
    >>> t = TIFF('astronaut3.tif', mode='w')
    >>> # Setup the first IFD.
    >>> for tag, value in tags.keys():
    ...     t[tag] = value
    >>> t[:] = image
    >>> # Finish off the first IFD and signal that there will be
    >>> # another.
    >>> t.write_directory()
    >>> # Setup the 2nd IFD. 
    >>> for tag, value in tags.keys():
    ...     t[tag] = value
    >>> t[:] = image
    >>> # Finish off the second IFD and signal that there will be
    >>> # another.
    >>> t.write_directory()
    >>> # Setup the 3rd IFD. 
    >>> for tag, value in tags.keys():
    ...     t[tag] = value
    >>> t[:] = image
    >>> del t

... create a BigTIFF...
==============================
Easy.  Just think about how libtiff does it with the ``TIFFOpen`` function, you
just use 'w8' instead of 'w' for the mode argument.

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

... view the majority of TIFFs easily?
======================================
Ok yes, that sounds strange.  But TIFF hasn't been described as
Thousands of Incompatible File Formats for no reason.  There's RGB,
Min-Is-Black, Min-Is-White, Palette, YCbCr/JPEG, Separated (CMYK), LogL,
LogLuv, and a whole slew of others.  Fortunately libtiff provides
a simple way to read all those photometric intrepretations named
above in a consistent and convenient manner and it's called the RGBA
interface.  Whether or not you should is up to you, but here's how
you can do it.  The following TIFF has a photometric interpretation
of YCbCr with old-JPEG compression. This is a case where you really
have no option but to use the convenience method, which only involves
setting the rgba property.  That translates into reading an image using
libtiff's RGBA interface.

.. plot:: pyplots/rgba.py
   :include-source:

... create a TIFF with subIFDs?
===============================
In this case, it helps to be a bit familiar with the workflow for
libtiff's C API.  Here, though, you need only supply the number of
IFDs you will be writing, then use set_subdirectory when you are finished
with the primary IFD.

We're going to be reusing a lot of tags, so we'll store them in a dictionary
and modify as needed.

    >>> import skimage.data
    >>> image = skimage.data.astronaut()
    >>> w, h, nz = image.shape
    >>> from spiff import TIFF, lib
    >>> t = TIFF('astronaut3.tif', mode='w')
    >>> tags = {
    ...     'Photometric': lib.Photometric.RGB,
    ...     'ImageWidth': w,
    ...     'ImageLength': h,
    ...     'TileWidth': int(w/2),
    ...     'TileLength': int(h/2),
    ...     'PlanarConfig': lib.PlanarConfig.CONTIG,
    ...     'BitsPerSample': 8,
    ...     'SamplesPerPixel': 3,
    ...     'Compression': lib.Compression.NONE,
    ... }
    >>> for tag, value in tags.items():
    ...     t[tag] = value


Now write the SubIFDs tag.  We will create two SubIFD images.

    >>> t['SubIFDs'] = 2

We have to finish by writing the primary IFD image, then we can move along to
the subIFDs.  Actually, we **MUST** move along to the subIFDs next.

    >>> t[:] = image
    >>> t.write_directory()

We will make the first IFD different by using LZW compression.

    >>> tags['Photometric'] = lib.Photometric.RGB
    >>> tags['Compression'] = lib.Compression.LZW
    >>> for tag, value in tags.items():
    ...     t[tag] = value
    >>> t[:] = image

And finally, position to the second subIFD and write that one using JPEG
compression and close the file.

    >>> t.write_directory()
    >>> tags['Photometric'] = lib.Photometric.YCBCR
    >>> tags['Compression'] = lib.Compression.JPEG
    >>> tags['JPEGColorMode'] = lib.JPEGColorMode.RGB
    >>> tags['JPEGQuality'] = 75
    >>> tags['YCbCrSubsampling'] = (1, 1)
    >>> for tag, value in tags.items():
    ...     t[tag] = value
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

We can reach each subIFD with the set_subdirectory method.

    >>> t.set_subdirectory(t['SubIFDs'][1])
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
