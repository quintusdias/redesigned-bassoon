import skimage.data
image = skimage.data.astronaut()   
w, h, nz = image.shape 
from spiff import TIFF, lib
t = TIFF('astronaut3.tif', mode='w8')   

# All three images will have the same tags, so set them all
# up in advance.
tags = {
    'Photometric': lib.Photometric.RGB,
    'ImageWidth': w,
    'ImageLength': h,
    'TileWidth': int(w / 2),
    'TileLength': int(h / 2),
    'PlanarConfig': lib.PlanarConfig.CONTIG,
    'BitsPerSample': 8,
    'SamplesPerPixel': 3,
    'Compression': lib.Compression.NONE,
}
t = TIFF('astronaut3.tif', mode='w')
# Setup the first IFD.
for tag, value in tags.items():
    t[tag] = value
t[:] = image
# Finish off the first IFD and signal that there will be
# another.
t.write_directory()
# Setup the 2nd IFD. 
for tag, value in tags.items():
    t[tag] = value
t[:] = image
# Finish off the second IFD and signal that there will be
# another.
t.write_directory()
# Setup the 3rd IFD. 
for tag, value in tags.items():
    t[tag] = value
t[:] = image
del t
