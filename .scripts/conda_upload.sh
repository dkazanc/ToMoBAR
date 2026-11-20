#!/bin/bash
set -euo pipefail

: "${ANACONDA_API_TOKEN:?Set the upload token before publishing}"

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Use a fresh output directory so only this build's packages can be uploaded.
export CONDA_BLD_PATH
CONDA_BLD_PATH="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/tomobar-conda.XXXXXX")"
conda build "$repo_root/conda-recipe" --no-anaconda-upload

shopt -s nullglob
artifacts=("$CONDA_BLD_PATH"/noarch/tomobar-*.conda "$CONDA_BLD_PATH"/noarch/tomobar-*.tar.bz2)
if ((${#artifacts[@]} == 0)); then
    echo "No ToMoBAR packages were produced" >&2
    exit 1
fi
for artifact in "${artifacts[@]}"; do
    echo "Uploading $artifact"
    anaconda -v --show-traceback -t "$ANACONDA_API_TOKEN" upload "$artifact" --force
done
