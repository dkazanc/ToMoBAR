#!/bin/bash

set -xe 
cp -rv "$RECIPE_DIR/../tomobar/cuda_kernels/" "$SRC_DIR/tomobar"

cd $SRC_DIR
python -m pip install .

#python -m pip install --target $CONDA_PREFIX/lib/python3.10/site-packages/ .
