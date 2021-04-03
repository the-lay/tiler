cd $(dirname "$file_name")/..
pwd

# build docs
pip install pdoc==6.4.2
pdoc -o docs -d google tiler

# add downsized teaser image to the docs directory
convert -resize 50% misc/teaser/tiler_teaser.png docs/tiler_teaser.png

# replace unnecessary index
rm docs/index.html
mv docs/tiler.html docs/index.html
