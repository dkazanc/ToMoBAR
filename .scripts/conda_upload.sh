#!/bin/bash

PKG_NAME=tomobar
USER=httomo-team
OS=noarch
CONDA_TOKEN=$(cat $HOME/secrets/my_secret.json)

mkdir ~/conda-bld
#conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld

$CONDA/bin/conda install conda-build
$CONDA/bin/conda install -c anaconda anaconda-client

export VERSION=$(date +%Y.%m)
$CONDA/bin/conda build .

# upload packages to conda
find $CONDA_BLD_PATH/$OS -name *.conda | while read file
do
    echo $file
    $CONDA/bin/anaconda -v --show-traceback -t $ANACONDA_API_TOKEN upload $file --force
done
