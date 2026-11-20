#!/usr/bin/env bash
# Build from maintained API pages and saved notebook outputs.
set -euo pipefail

docs_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python -m sphinx -a -E -W --keep-going -b html \
    "$docs_dir/source" "$docs_dir/build"
