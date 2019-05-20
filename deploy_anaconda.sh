#!/bin/bash
# this script uses the ANACONDA_TOKEN env var. 
# to create a token:
# >>> anaconda login
# >>> anaconda auth -c -n travis --max-age 307584000 --url https://anaconda.org/USERNAME/PACKAGENAME --scopes "api:write api:read"
set -e

echo "Converting conda package..."
conda convert --platform all /home/travis/miniconda/envs/test-environment/conda-bld/linux-64/tomobar-*.tar.bz2 --output-dir conda-bld/

echo "Deploying to Anaconda.org..."
anaconda -t $CONDA_UPLOAD_TOKEN upload conda-bld/**/tomobar-*.tar.bz2

echo "Successfully deployed to Anaconda.org."
exit 0
