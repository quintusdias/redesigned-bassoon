from matplotlib import pyplot as plt
from spiff import TIFF, data
t = TIFF(data.zackthecat)
t.rgba = True
img = t[:]
plt.imshow(img)
plt.show()
