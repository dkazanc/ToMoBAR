#!/bin/bash
# Requires Cython, install it first: conda install cython
echo "Building ToMoBAR software using CMake"
rm -r build
mkdir build
cd build
#make clean
export VERSION=`date +%Y.%m`
# install Python modules with CMAKE
cmake ../ -DBUILD_PYTHON_WRAPPER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make install
############### Python(linux)###############
cp install/lib/libtomobar.so install/python/tomobar/supp/
cd install/python
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../lib
spyder --new-instance
