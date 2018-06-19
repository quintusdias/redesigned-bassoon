# Standard library imports ...
import ctypes

TAGS = {
    'SubFileType': {
        'number': 254,
        'type': ctypes.c_uint16,
    },
    'OSubFileType': {
        'number': 255,
        'type': ctypes.c_uint16,
    },
    'ImageWidth': {
        'number': 256,
        'type': ctypes.c_uint32,
    },
    'ImageLength': {
        'number': 257,
        'type': ctypes.c_uint32,
    },
    'BitsPerSample': {
        'number': 258,
        'type': ctypes.c_uint16,
    },
    'Compression': {
        'number': 259,
        'type': ctypes.c_uint16,
    },
    'Photometric': {
        'number': 262,
        'type': ctypes.c_uint16,
    },
    'Threshholding': {
        'number': 263,
        'type': ctypes.c_uint16,
    },
    'CellWidth': {
        'number': 264,
        'type': ctypes.c_uint16,
    },
    'CellLength': {
        'number': 265,
        'type': ctypes.c_uint16,
    },
    'FillOrder': {
        'number': 266,
        'type': ctypes.c_uint16,
    },
    'DocumentName': {
        'number': 269,
        'type': ctypes.c_char_p,
    },
    'ImageDescription': {
        'number': 270,
        'type': ctypes.c_char_p,
    },
    'Make': {
        'number': 271,
        'type': ctypes.c_char_p,
    },
    'Model': {
        'number': 272,
        'type': ctypes.c_char_p,
    },
    'StripOffsets': {
        'number': 273,
        'type': (ctypes.c_uint32, ctypes.c_uint64),
    },
    'Orientation': {
        'number': 274,
        'type': ctypes.c_uint16,
    },
    'SamplesPerPixel': {
        'number': 277,
        'type': ctypes.c_uint16,
    },
    'RowsPerStrip': {
        'number': 278,
        'type': ctypes.c_uint16,
    },
    'StripByteCounts': {
        'number': 279,
        'type': None,
    },
    'MinSampleValue': {
        'number': 280,
        'type': ctypes.c_uint16,
    },
    'MaxSampleValue': {
        'number': 281,
        'type': ctypes.c_uint16,
    },
    'XResolution': {
        'number': 282,
        'type': ctypes.c_double,
    },
    'YResolution': {
        'number': 283,
        'type': ctypes.c_double,
    },
    'PlanarConfig': {
        'number': 284,
        'type': ctypes.c_uint16,
    },
    'PageName': {
        'number': 285,
        'type': ctypes.c_char_p,
    },
    'XPosition': {
        'number': 286,
        'type': ctypes.c_double,
    },
    'YPosition': {
        'number': 287,
        'type': ctypes.c_double,
    },
    'FreeOffsets': {
        'number': 288,
        'type': ctypes.c_uint32,
    },
    'FreeByteCounts': {
        'number': 289,
        'type': ctypes.c_uint32,
    },
    'GrayResponseUnit': {
        'number': 290,
        'type': ctypes.c_uint16,
    },
    'GrayResponseCurve': {
        'number': 291,
        'type': None,
    },
    'T4Options': {
        'number': 292,
        'type': None,
    },
    'T6Options': {
        'number': 293,
        'type': None,
    },
    'ResolutionUnit': {
        'number': 296,
        'type': ctypes.c_uint16,
    },
    'PageNumber': {
        'number': 297,
        'type': (ctypes.c_uint16, ctypes.c_uint16),
    },
    'TransferFunction': {
        'number': 301,
        'type': None,
    },
    'Software': {
        'number': 305,
        'type': ctypes.c_char_p,
    },
    'Datetime': {
        'number': 306,
        'type': ctypes.c_char_p,
    },
    'Artist': {
        'number': 315,
        'type': ctypes.c_char_p,
    },
    'HostComputer': {
        'number': 316,
        'type': ctypes.c_char_p,
    },
    'Predictor': {
        'number': 317,
        'type': ctypes.c_uint16,
    },
    'WhitePoint': {
        'number': 318,
        'type': ctypes.c_double,
    },
    'PrimaryChromaticities': {
        'number': 319,
        'type': None,
    },
    'Colormap': {
        'number': 320,
        'type': (ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16),
    },
    'HalfToneHints': {
        'number': 321,
        'type': ctypes.c_uint16,
    },
    'TileWidth': {
        'number': 322,
        'type': ctypes.c_uint32,
    },
    'TileLength': {
        'number': 323,
        'type': ctypes.c_uint32,
    },
    'TileOffsets': {
        'number': 324,
        'type': None,
    },
    'TileByteCounts': {
        'number': 325,
        'type': None,
    },
    'BadFaxLines': {
        'number': 326,
        'type': None,
    },
    'CleanFaxData': {
        'number': 327,
        'type': None,
    },
    'ConsecutiveBadFaxLines': {
        'number': 328,
        'type': None,
    },
    'SubIFDs': {
        'number': 330,
        'type': None,
    },
    'InkSet': {
        'number': 332,
        'type': ctypes.c_uint16,
    },
    'InkNames': {
        'number': 333,
        'type': ctypes.c_char_p,
    },
    'NumberOfInks': {
        'number': 334,
        'type': ctypes.c_uint16,
    },
    'DotRange': {
        'number': 336,
        'type': None,
    },
    'TargetPrinter': {
        'number': 337,
        'type': ctypes.c_uint16,
    },
    'ExtraSamples': {
        'number': 338,
        'type': ctypes.c_uint16,
    },
    'SampleFormat': {
        'number': 339,
        'type': ctypes.c_uint16,
    },
    'SMinSampleValue': {
        'number': 340,
        'type': ctypes.c_double,
    },
    'SMaxSampleValue': {
        'number': 341,
        'type': ctypes.c_double,
    },
    'TransferRange': {
        'number': 342,
        'type': None,
    },
    'ClipPath': {
        'number': 343,
        'type': None,
    },
    'XClipPathUnits': {
        'number': 344,
        'type': None,
    },
    'YClipPathUnits': {
        'number': 345,
        'type': None,
    },
    'Indexed': {
        'number': 346,
        'type': None,
    },
    'JPEGTables': {
        'number': 347,
        'type': None,
    },
    'OPIProxy': {
        'number': 351,
        'type': None,
    },
    'GlobalParametersIFD': {
        'number': 400,
        'type': None,
    },
    'ProfileType': {
        'number': 401,
        'type': None,
    },
    'FaxProfile': {
        'number': 402,
        'type': ctypes.c_uint8,
    },
    'CodingMethods': {
        'number': 403,
        'type': None,
    },
    'VersionYear': {
        'number': 404,
        'type': None,
    },
    'ModeNumber': {
        'number': 405,
        'type': None,
    },
    'Decode': {
        'number': 433,
        'type': None,
    },
    'DefaultImageColor': {
        'number': 434,
        'type': None,
    },
    'JPEGProc': {
        'number': 512,
        'type': None,
    },
    'JPEGInterchangeFormat': {
        'number': 513,
        'type': None,
    },
    'JPEGInterchangeFormatLength': {
        'number': 514,
        'type': None,
    },
    'JPEGRestartInterval': {
        'number': 515,
        'type': None,
    },
    'JPEGLosslessPredictors': {
        'number': 517,
        'type': None,
    },
    'JPEGPointTransforms': {
        'number': 518,
        'type': None,
    },
    'JPEGQTables': {
        'number': 519,
        'type': None,
    },
    'JPEGDCTables': {
        'number': 520,
        'type': None,
    },
    'JPEGACTables': {
        'number': 521,
        'type': None,
    },
    'YCbCrCoefficients': {
        'number': 529,
        'type': (ctypes.c_float, ctypes.c_float, ctypes.c_float),
    },
    'YCbCrSubsampling': {
        'number': 530,
        'type': (ctypes.c_uint16, ctypes.c_uint16),
    },
    'YCbCrPositioning': {
        'number': 531,
        'type': ctypes.c_uint16,
    },
    'ReferenceBlackWhite': {
        'number': 532,
        'type': (ctypes.c_float, ctypes.c_float, ctypes.c_float,
                 ctypes.c_float, ctypes.c_float, ctypes.c_float),
    },
    'StripRowCounts': {
        'number': 559,
        'type': None,
    },
    'XMP': {
        'number': 700,
        'type': ctypes.c_uint8,
    },
    'ImageID': {
        'number': 32781,
        'type': None,
    },
    'Datatype': {
        'number': 32996,
        'type': None,
    },
    'WANGAnnotation': {
        'number': 32932,
        'type': None,
    },
    'ImageDepth': {
        'number': 32997,
        'type': None,
    },
    'TileDepth': {
        'number': 32998,
        'type': None,
    },
    'Copyright': {
        'number': 33432,
        'type': ctypes.c_char_p,
    },
    'MDFile': {
        'number': 33445,
        'type': None,
    },
    'MDScalePixel': {
        'number': 33446,
        'type': None,
    },
    'MDColorTable': {
        'number': 33447,
        'type': None,
    },
    'MDLabName': {
        'number': 33448,
        'type': None,
    },
    'MDSampleInfo': {
        'number': 33449,
        'type': None,
    },
    'MdPrepDate': {
        'number': 33450,
        'type': None,
    },
    'MDPrepTime': {
        'number': 33451,
        'type': None,
    },
    'MDFileUnits': {
        'number': 33452,
        'type': None,
    },
    'ModelPixelScale': {
        'number': 33550,
        'type': None,
    },
    'IPTC': {
        'number': 33723,
        'type': None,
    },
    'INGRPacketData': {
        'number': 33918,
        'type': None,
    },
    'INGRFlagRegisters': {
        'number': 33919,
        'type': None,
    },
    'IRASbTransformationMatrix': {
        'number': 33920,
        'type': None,
    },
    'ModelTiePoint': {
        'number': 33922,
        'type': None,
    },
    'ModelTransformation': {
        'number': 34264,
        'type': None,
    },
    'Photoshop': {
        'number': 34377,
        'type': None,
    },
    'ExifIFD': {
        'number': 34665,
        'type': ctypes.c_int32,
    },
    'ICCProfile': {
        'number': 34675,
        'type': None,
    },
    'ImageLayer': {
        'number': 34732,
        'type': None,
    },
    'GeoKeyDirectory': {
        'number': 34735,
        'type': None,
    },
    'GeoDoubleParams': {
        'number': 34736,
        'type': None,
    },
    'GeoASCIIParams': {
        'number': 34737,
        'type': None,
    },
    'GPSIFD': {
        'number': 34853,
        'type': None,
    },
    'HYLAFAXRecvParams': {
        'number': 34908,
        'type': None,
    },
    'HYLAFAXSubAddress': {
        'number': 34909,
        'type': None,
    },
    'HYLAFAXRecvTime': {
        'number': 34910,
        'type': None,
    },
    'ImageSourceData': {
        'number': 37724,
        'type': None,
    },
    'InteroperabilityIFD': {
        'number': 40965,
        'type': None,
    },
    'GDAL_Metadata': {
        'number': 42112,
        'type': None,
    },
    'GDAL_NoData': {
        'number': 42113,
        'type': None,
    },
    'OCEScanJobDescription': {
        'number': 50215,
        'type': None,
    },
    'OCEApplicationSelector': {
        'number': 50216,
        'type': None,
    },
    'OCEIdentificationNumber': {
        'number': 50217,
        'type': None,
    },
    'OCEImageLogicCharacteristics': {
        'number': 50218,
        'type': None,
    },
    'DNGVersion': {
        'number': 50706,
        'type': None,
    },
    'DNGBackwardVersion': {
        'number': 50707,
        'type': None,
    },
    'UniqueCameraModel': {
        'number': 50708,
        'type': None,
    },
    'LocalizedCameraModel': {
        'number': 50709,
        'type': None,
    },
    'CFAPlaneColor': {
        'number': 50710,
        'type': None,
    },
    'CFALayout': {
        'number': 50711,
        'type': None,
    },
    'LinearizationTable': {
        'number': 50712,
        'type': None,
    },
    'BlackLevelRepeatDim': {
        'number': 50713,
        'type': None,
    },
    'BlackLevel': {
        'number': 50714,
        'type': None,
    },
    'BlackLevelDeltaH': {
        'number': 50715,
        'type': None,
    },
    'BlackLevelDeltaV': {
        'number': 50716,
        'type': None,
    },
    'WhiteLevel': {
        'number': 50717,
        'type': None,
    },
    'DefaultScale': {
        'number': 50718,
        'type': None,
    },
    'DefaultCropOrigin': {
        'number': 50719,
        'type': None,
    },
    'DefaultCropSize': {
        'number': 50720,
        'type': None,
    },
    'ColorMatrix1': {
        'number': 50721,
        'type': None,
    },
    'ColorMatrix2': {
        'number': 50722,
        'type': None,
    },
    'CameraCalibration1': {
        'number': 50723,
        'type': None,
    },
    'CameraCalibration2': {
        'number': 50724,
        'type': None,
    },
    'ReductionMatrix1': {
        'number': 50725,
        'type': None,
    },
    'ReductionMatrix2': {
        'number': 50726,
        'type': None,
    },
    'AnalogBalance': {
        'number': 50727,
        'type': None,
    },
    'AsShotNeutral': {
        'number': 50728,
        'type': None,
    },
    'AsShotWhiteXY': {
        'number': 50729,
        'type': None,
    },
    'BaselineExposure': {
        'number': 50730,
        'type': None,
    },
    'BaselineNoise': {
        'number': 50731,
        'type': None,
    },
    'BaselineSharpness': {
        'number': 50732,
        'type': None,
    },
    'BayerGreenSplit': {
        'number': 50733,
        'type': None,
    },
    'LinearResponseLimit': {
        'number': 50734,
        'type': None,
    },
    'CameraSerialNumber': {
        'number': 50735,
        'type': None,
    },
    'LensInfo': {
        'number': 50736,
        'type': None,
    },
    'ChromaBlurRadius': {
        'number': 50737,
        'type': None,
    },
    'AntiAliasStrength': {
        'number': 50738,
        'type': None,
    },
    'DNGPrivateData': {
        'number': 50740,
        'type': None,
    },
    'MakerNoteSafety': {
        'number': 50741,
        'type': None,
    },
    'CalibrationIllumintant1': {
        'number': 50778,
        'type': None,
    },
    'CalibrationIllumintant2': {
        'number': 50779,
        'type': None,
    },
    'BestQualityScale': {
        'number': 50780,
        'type': None,
    },
    'AliasLayerMetadata': {
        'number': 50784,
        'type': None,
    },
    'TIFF_RSID': {
        'number': 50908,
        'type': None,
    },
    'GEO_Metadata': {
        'number': 50909,
        'type': None,
    },
    'JPEGQuality': {
        'number': 65537,
        'type': ctypes.c_int32,
    },
    'JPEGColorMode': {
        'number': 65538,
        'type': ctypes.c_int32,
    },
}

# We need the reverse mapping as well.
tagnum2name = {value['number']: key for key, value in TAGS.items()}
