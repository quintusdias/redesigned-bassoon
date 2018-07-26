# Standard library imports ...
from distutils.core import setup
from distutils.core import Extension
import pathlib
import platform
import sys

# Third party library imports ...
from Cython.Build import cythonize

if ((('Anaconda' in sys.version) or
     ('Continuum Analytics, Inc.' in sys.version) or
     ('packaged by conda-forge' in sys.version))):
    # If Anaconda, then tiff may have been installed via conda.
    if platform.system() in ['Linux', 'Darwin']:
        root = pathlib.Path(sys.executable).parents[1]
    elif platform.system() == 'Windows':
        root = pathlib.Path(sys.executable).parents[0]
    include_dirs = [str(root / 'include')]
    lib_dirs = [str(root / 'lib')]
elif platform.system() == 'Darwin':
    # MacPorts?  HomeBrew?
    include_dirs = ['/opt/local/include', '/usr/local/include']
    lib_dirs = ['/opt/local/lib', '/usr/local/lib']
else:
    include_dirs = ['/usr/include']
    lib_dirs = ['/usr/lib', '/usr/lib64']

extension = Extension('spiff._cytiff',
                      sources=['spiff/_cytiff.pyx'],
                      libraries=['tiff', 'jpeg', 'lzma', 'z'],
                      include_dirs=include_dirs,
                      library_dirs=lib_dirs,
                      language="c++")

kwargs = {
    'name': 'Spiff',
    'description': 'Tools for accessing TIFFs',
    'long_description': open('README.md').read(),
    'author': 'John Evans',
    'author_email': 'john.g.evans.ne@gmail.com',
    'url': 'https://github.com/quintusdias/redesigned-bassoon',
    'packages': ['spiff', 'spiff.data', 'spiff.lib'],
    'package_data': {'spiff': ['data/*.tif']},
    'license': 'MIT',
    'ext_modules': cythonize([extension]),
    'test_suite': 'spiff.tests',
    'install_requires': ['setuptools'],
    'version': '0.0.1',
    'classifiers': [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: CPython",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows :: Windows XP",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ]
}

setup(**kwargs)
