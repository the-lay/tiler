# tiler

![Demo gif](misc/teaser/tiler_teaser.png)

![CI](https://github.com/the-lay/tiler/workflows/CI/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/the-lay/tiler/badge.svg?branch=master)](https://coveralls.io/github/the-lay/tiler?branch=master)
[![PyPI version](https://badge.fury.io/py/tiler.svg)](https://badge.fury.io/py/tiler)

### Please note: work in progress, things might change and/or break!

This package provides tools for N-dimensional tiling (patch extraction)
and subsequent merging with built-in tapering (window) function support.

This is especially helpful for various heavy-processing tasks such as
semantic segmentation in deep learning, especially in domains where
images do not fit into GPU memory (2D hyperspectral satellite images, 
2D whole slide images, 2D videos, 3D tomographic data etc.).

Implemented features
-------------
 - Data reader agnostic: works with numpy arrays
 - N-dimensional array tiling
   (note: currently tile shape must have the same number of dimensions as the array)
 - Optional in-place tiling (without creating copies)
 - Supports channel dimension: dimension that will not be tiled
 - Supports batching of tiles
 - Overlapping support: you can specify tile percentage or directly overlap size
 - Window functions: Merger accepts weights for the tile as an array or a scipy window
 - Convenient access to the tiles: with an iterator or a separate getter
 - Easy merging to the full size: just add the processed tile to the Merger

Quick start
------------
This is an example of basic functionality.  
You can also find more examples in [examples/](examples).  
For more Tiler and Merger functionality, please check documentation.

```python
import numpy as np
from tiler import Tiler, Merger

image = np.random.random((3, 1920, 1080))

# Setup tiling parameters
tiler = Tiler(image_shape=image.shape,
              tile_shape=(3, 250, 250),
              channel_dimension=2)

# You can access tiles with an iterator
for tile_id, tile in tiler(image):
    print(f'Tile {tile_id} out of {len(tiler)} tiles.')

# You can access tiles individually
tile_3 = tiler.get_tile(image, 3)

# You can access tiles in batches
tiles_in_batches = [batch for _, batch in tiler(image, batch_size=10, drop_last=True)]

# Setup merger object with constant window
merger = Merger(tiler, window='boxcar')

# Merge one by one
for tile_id, tile in tiler(image):
   merger.add(tile_id, some_processing_fn(tile))
final_image = merger.merge(unpad=True)

# Merge in batches
merger.reset()
for batch_id, batch in tiler(image, batch_size=10):
   merger.add_batch(batch_id, 10, batch)
final_image_batches = merger.merge(unpad=True)

final_image.shape, final_image_batches.shape
>>> (3, 1920, 1080), (3, 1920, 1080)
```
 
Installation
-------------
The latest release is available through pip:

```
pip install tiler 
 ```

Alternatively, you can clone the repository and install it manually:

```
git clone git@github.com:the-lay/tiler.git
cd tiler
pip install .
```

Roadmap
------------
 - Proper documentation
 - Easy generation of tiling for specific window in mind
   (i.e. so that every element has the window weight sum of 1.0)
 - Add border windows generation (like in Pielawski et. al - see references))
 - PyTorch Tensors support
   - merging on GPU like in pytorch-toolbelt?
 - More examples
 - Implement windows functions and remove scipy dependency
   (we need only a couple of functions that generate windows)
 - PyTorch Dataset class convenience wrapper?
 - Arbitrary sized tiles (m-dim window over n-dim array, m <= n)?
    - [Some leads here](https://stackoverflow.com/questions/45960192/using-numpy-as-strided-function-to-create-patches-tiles-rolling-or-sliding-w)
 - Optional augmentation modes for smoother segmentations?
    - D4 rotation group
    - Mirroring
 - Benchmark with plain for loops, determine overhead
 
Motivation & other tiling/patching packages
-------------
I work on semantic segmentation of patched 3D data and
I often found myself reusing tiling functions that I wrote for previous projects.
No existing libraries listed below fit my use case, so that's why I wrote `tiler`.

However, other libraries might fit you better than `tiler`:
 - [vfdev-5/ImageTilingUtils](https://github.com/vfdev-5/ImageTilingUtils)
    - Minimalistic image reader agnostic 2D tiling tools

 - [BloodAxe/pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt#inference-on-huge-images)
    - Powerful PyTorch toolset that has 2D image tiling and on-GPU merger

 - [Vooban/Smoothly-Blend-Image-Patches](https://github.com/Vooban/Smoothly-Blend-Image-Patches)
    - Mirroring and D4 rotations data (8-fold) augmentation with squared spline window function for 2D images

 - [samdobson/image_slicer](https://github.com/samdobson/image_slicer)
    - Slicing and merging 2D image into N equally sized tiles

 - [dovahcrow/patchify.py](https://github.com/dovahcrow/patchify.py)
    - Tile and merge 2D, 3D images defined by tile shapes and step between tiles
   
 - Do you know any other similar packages?
    - [Please make a PR](https://github.com/the-lay/tiler/pulls) or [open a new issue](https://github.com/the-lay/tiler/issues).

Moreover, some approaches have been described in the literature:
 - [Introducing Hann windows for reducing edge-effects in patch-based image segmentation](https://doi.org/10.1371/journal.pone.0229839
), Pielawski and Wählby, March 2020







<!-- for later
For more examples, please see examples/ folder
```python

 ```

Benchmarks
-------------
 Benchmarks?
 

Examples
-------------
https://github.com/BloodAxe/pytorch-toolbelt#inference-on-huge-images
https://github.com/BloodAxe/pytorch-toolbelt/blob/master/pytorch_toolbelt/inference/tiles.py

https://github.com/vfdev-5/ImageTilingUtils

https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/smooth_tiled_predictions.py

for windows:
https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python

https://en.wikipedia.org/wiki/Window_function#A_list_of_window_functions
https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/windows/windows.py
https://gist.github.com/npielawski/7e77d23209a5c415f55b95d4aba914f6

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0229839#pone.0229839.ref005
https://arxiv.org/pdf/1803.02786.pdf
-->
