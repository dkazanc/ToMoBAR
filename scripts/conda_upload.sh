#!/bin/sh
PKG_NAME=tomobar
USER=dkazanc
OS=linux-64
CONDA_TOKEN=$(cat $HOME/secrets/my_secret.json)
export VERSION=`date +%Y.%m`

array=( 3.5 3.6 3.7 )

mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld

# building conda packages
for i in "${array[@]}"
do
	conda-build . --python $i $pkg
done

# upload packages to conda
find $CONDA_BLD_PATH/$OS -name *.tar.bz2 | while read file
do
    echo $file
    $CONDA/bin/anaconda -v --show-traceback --token $CONDA_TOKEN upload $file --force
done
