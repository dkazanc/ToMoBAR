#!/bin/bash

PKG_NAME=tomobar
USER=dkazanc
OS=linux-64
CONDA_TOKEN=$(cat $HOME/.secrets/my_secret.json)

mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
export VERSION=`date +%Y.%m`
conda install --yes anaconda-client
conda build . --numpy=1.20 --python=3.9
conda build . --numpy=1.21 --python=3.9
conda build . --numpy=1.22 --python=3.9
conda build . --numpy=1.20 --python=3.10
conda build . --numpy=1.21 --python=3.10
conda build . --numpy=1.22 --python=3.10

# upload packages to conda
find $CONDA_BLD_PATH/$OS -name *.tar.bz2 | while read file
do
    echo $file
    $CONDA/bin/anaconda -v --show-traceback --token $CONDA_TOKEN upload $file --force
done
