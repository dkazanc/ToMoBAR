#!/usr/bin/env python
from distutils.core import setup
#import os
#import sys

setup(
    name="fista-tomo",
    version='0.1.1',
    packages=['fista', 'fista.tomo'],

    # metadata for upload to PyPI
    author="Daniil Kazantsev",
    author_email="daniil.kazantsev@diamond.ac.uk",
    description='FISTA iterative reconstruction algorithm for tomography',
    license="GPL v3",
    keywords="Python Framework",
    url="https://github.com/dkazanc/FISTA-tomo",  
)
