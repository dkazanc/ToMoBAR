#!/bin/bash

PKG_NAME=tomobar
USER=dkazanc
OS=linux-64
CONDA_TOKEN=$(cat $HOME/.secrets/my_secret.json)

mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld

$CONDA/bin/conda install -c conda-forge conda-build
$CONDA/bin/conda install -c conda-forge anaconda-client
$CONDA/bin/conda install -c conda-forge mamba

#export VERSION=$(date +%Y.%m)
#conda build .

for python_ver in 3.9; do   
    for numpy_ver in 1.21; do
        export VERSION=`date +%Y.%m`"_py"$python_ver"_np"$numpy_ver
        mamba build . --numpy $numpy_ver --python $python_ver
   done
done

# upload packages to conda
find $CONDA_BLD_PATH/$OS -name *.tar.bz2 | while read file
do
    echo $file
    $CONDA/bin/anaconda -v --show-traceback --token $CONDA_TOKEN upload $file --force
done
