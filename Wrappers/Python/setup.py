#!/usr/bin/env python
from distutils.core import setup
#import os
#import sys

setup(
    name="tomorec",
    version='0.1.2',
    packages=['tomorec', 'tomorec.supp'],

    # metadata for upload to PyPI
    author="Daniil Kazantsev",
    author_email="daniil.kazantsev@diamond.ac.uk",
    description="MATLAB/Python library of tomographic reconstruction methods: direct and iterative",
    license="GPL v3",
    keywords="Python Framework",
    url="https://github.com/dkazanc/TomoRec",
)
