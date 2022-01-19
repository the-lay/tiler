cd "$(dirname "$file_name")"/.. || exit

# build docs
pdoc -o docs -d google tiler

# add downsized teaser image to the docs directory
convert -resize 50% misc/teaser/tiler_teaser.png docs/tiler_teaser.png

# replace unnecessary index
rm docs/index.html
mv docs/tiler.html docs/index.html

# remove unused search index
rm -f docs/search.js
