# Third party library imports ...
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extension = Extension('spiff._cytiff',
                      sources=['spiff/_cytiff.pyx'],
                      libraries=['tiff', 'jpeg', 'lzma', 'z'],
                      include_dirs=['/opt/local/include'],
                      library_dir=['/opt/local/lib'])

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
