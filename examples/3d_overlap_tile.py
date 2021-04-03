# 3D grayscale overlap-tile stratefy tiling/merging example
#
# "This strategy allows the seamless segmentation of arbitrarily large images by an overlap-tile strategy.
# To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring
# the input image. This tiling strategy is important to apply the network to large images,
# since otherwise the resolution would be limited by the GPU memory." - Ronneberger et al 2015, U-Net paper

# We will use napari to inspect 3D volumes
import numpy as np
from tiler import Tiler, Merger
import napari

# Example "checkerboard"-like volume with some variation for visualization
# https://stackoverflow.com/a/51715491
volume = (np.indices((150, 462, 462)).sum(axis=0) % 50).astype(np.float32)
volume[:75] *= np.linspace(3, 10, 75)[:, None, None]
volume[75:] *= np.linspace(10, 3, 75)[:, None, None]

# Let's assume we want to use tiles of size 48x48x48 and only the middle 20x20x20 for the final image
# That means we need to pad the image by 14 from each side
# To extrapolate missing context let's use reflect mode
padded_volume = np.pad(volume, 14, mode='reflect')

# Specifying tiling
# The overlap should be 28 voxels
tiler = Tiler(data_shape=padded_volume.shape,
              tile_shape=(48, 48, 48),
              overlap=(28, 28, 28))

# Window function for merging
window = np.zeros((48, 48, 48))
window[14:-14, 14:-14, 14:-14] = 1

# Specifying merging
merger = Merger(tiler=tiler, window=window)

# Let's define a function that will be applied to each tile
# For this example, let's black out the sides that should be "cropped" by window function
# as a way to confirm that only the middle parts are being merged
def process(patch: np.ndarray) -> np.ndarray:
    patch[:14, :, :] = 0
    patch[-14:, :, :] = 0
    patch[:, :14, :] = 0
    patch[:, -14:, :] = 0
    patch[:, :, :14] = 0
    patch[:, :, -14:] = 0
    return patch

# Iterate through all the tiles and apply the processing function and merge everything back
for tile_id, tile in tiler(padded_volume, progress_bar=True):
    processed_tile = process(tile)
    merger.add(tile_id, processed_tile)

final_volume = merger.merge()
final_unpadded_volume = final_volume[14:-14, 14:-14, 14:-14]

# Show all the
with napari.gui_qt():
    v = napari.Viewer()
    v.add_image(volume, name='Original volume')
    v.add_image(padded_volume, name='Padded volume')
    v.add_image(final_volume, name='Final volume')
    v.add_image(final_unpadded_volume, name='Final unpadded volume')
    v.add_image(merger.weights_sum, name='Merger weights sum')
    v.add_image(merger.data_visits, name='Merger data visits')
