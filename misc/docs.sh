#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# build docs
pdoc -o docs -d google src/tiler

# add downsized teaser image to the docs directory
magick misc/teaser/tiler_teaser.png -resize 50% docs/tiler_teaser.png

# replace unnecessary index
rm docs/index.html
mv docs/tiler.html docs/index.html

# remove unused search index
rm -f docs/search.js
