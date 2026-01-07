#!/bin/bash

PKG_NAME=tomobar
USER=httomo-team
OS=noarch

mkdir ~/conda-bld
export CONDA_BLD_PATH=~/conda-bld

$CONDA/bin/conda install conda-build
$CONDA/bin/conda install -c anaconda anaconda-client

export VERSION=1.0.0
$CONDA/bin/conda build .

# upload packages to conda
find $CONDA_BLD_PATH/$OS -name *.conda | while read file
do
    echo $file
    $CONDA/bin/anaconda -v --show-traceback -t $ANACONDA_API_TOKEN upload $file --force
done
