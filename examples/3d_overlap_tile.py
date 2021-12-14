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

# Let's assume we want to use tiles of size 48x48x48 and only the middle 20x20x20 contribute to the final image
# The overlap then should be 28 (48-20) voxels
tiler = Tiler(
    data_shape=volume.shape,
    tile_shape=(48, 48, 48),
    overlap=(28, 28, 28),
)

# Calculate and apply extra padding, as well as adjust tiling parameters
new_shape, padding = tiler.calculate_padding()
tiler.recalculate(data_shape=new_shape)
padded_volume = np.pad(volume, padding, mode="reflect")

# Specifying merging parameters
# You can define overlap-tile window explicitly, i.e.
# >>> window = np.zeros((48, 48, 48))
# >>> window[14:-14, 14:-14, 14:-14] = 1
# >>> merger = Merger(tiler=tiler, window=window)
# or you can use window="overlap-tile"
# it will automatically calculate such window based on tiler.overlap and applied padding
merger = Merger(tiler=tiler, window="overlap-tile")

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


# Apply the processing function to each tile and add it to the merger
for tile_id, tile in tiler(padded_volume, progress_bar=True):
    processed_tile = process(tile)
    merger.add(tile_id, processed_tile)

# Merge processed tiles
final_volume = merger.merge(extra_padding=padding)

print(f"Sanity check: {np.all(volume == final_volume)}")

# Show all the produced volumes
v = napari.Viewer()
# v.add_image(merger.window, name="window")
# v.add_image(tile, name="tile")
# v.add_image(processed_tile, name="processed_tile")
v.add_image(volume, name="Original volume")
v.add_image(padded_volume, name="Padded volume")
v.add_image(final_volume, name="Final merged volume")
v.add_image(merger.weights_sum, name="Merger weights sum")
v.add_image(merger.data_visits, name="Merger data visits")
v.show(block=True)
