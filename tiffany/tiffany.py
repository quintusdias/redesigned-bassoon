from . import lib

class TIFF(object):
    def __init__(self, filename):
        self.fp = lib.open(filename)

        tags = {}
        tags['width'] = lib.getField(self.fp, 'width')
        tags['height'] = lib.getField(self.fp, 'length')
        tags['sampleformat'] = lib.getFieldDefaulted(self.fp, 'sampleformat')
        tags['bitspersample'] = lib.getFieldDefaulted(self.fp, 'bitspersample')
        self.tags = tags

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if idx.start is None and idx.stop is None and idx.step is None:
                # case is [:]
                img = lib.readRGBAImage(self.fp, width=self.tags['width'],
                                        height=self.tags['height'])
                return img

