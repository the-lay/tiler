# tiler
[![PyPI version](https://badge.fury.io/py/tiler.svg)](https://badge.fury.io/py/tiler)

### Please note: work in progress! Until a stable release, please avoid using this package in production!

This package provides tools for N-dimensional tiling (patch extraction)
and subsequent merging with tapering (window) functions.

This is especially helpful for semantic segmentation tasks in deep learning,
where we often have to work with images that do not fit into GPU memory
(2D hyperspectral satellite images, 2D whole slide images, 2D videos, 3D tomographic data etc.).

Implemented features
-------------
 - Data reader agnostic: works with numpy arrays
 - Optimized to avoid unnecessary memory copies: numpy views for all tiles, except border tiles that require padding
 - N-dimensional array tiling (but for now tiles must have the same number of dimensions as the array)
 - Supports channel dimension: dimension that will not be tiled
 - Overlapping support: you can specify tile percentage or directly overlap size
 - Access individual tiles with consistent indexing or with a convenience iterator
 - Merging tiles back into a full array with optional un-padding to the original shape
 - Merging supports scipy window functions
 
Roadmap
------------
 - Proper documentation
 - Teaser image for github
 - Batched tiles
 - Implement overlap-tile strategy
 - Ability to generate tiling for specific window in mind (=> so that every element has the window weight sum of 1.0)
 - Add border windows generation (like in Pielawski et. al - see references))
 - PyTorch Tensors support
   - merging on GPU like in pytorch-toolbelt?
 - More examples
 - Implement windows functions and remove scipy dependency (we need only a couple of functions that generate windows)
 - PyTorch Dataset class convenience wrapper?
 - Arbitrary sized tiles (m-dim window over n-dim array, m <= n)?
    - [Some leads here](https://stackoverflow.com/questions/45960192/using-numpy-as-strided-function-to-create-patches-tiles-rolling-or-sliding-w)
 - Optional augmentation modes for smoother segmentations?
    - D4 rotation group
    - Mirroring
 - Benchmark with plain for loops, determine overhead
 
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
 
Examples
-------------
For now, only the one `examples/tiler_example.py`




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
    
 - Do you know any other similar packages?
    - [Please make a PR](https://github.com/the-lay/tiler/pulls)
    - or [open a new issue](https://github.com/the-lay/tiler/issues).
   
Academic references
-------------
[Introducing Hann windows for reducing edge-effects in patch-based image segmentation](https://doi.org/10.1371/journal.pone.0229839
), Pielawski and Wählby, March 2020
