# Third party library imports ...
from setuptools import setup

kwargs = {
    'name': 'Tiffany',
    'description': 'Tools for accessing TIFFs',
    'long_description': open('README.md').read(),
    'author': 'John Evans',
    'author_email': 'john.g.evans.ne@gmail.com',
    'url': 'https://github.com/quintusdias/redesigned-bassoon',
    'packages': ['tiffany', 'tiffany.data', 'tiffany.lib'],
    'package_data': {'tiffany': ['data/*.tif']},
    'license': 'MIT',
    'test_suite': 'tiffany.tests',
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
