# 2D RGB overlap-tile strategy tiling/merging example
#
# "This strategy allows the seamless segmentation of arbitrarily large images by an overlap-tile strategy.
# To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring
# the input image. This tiling strategy is important to apply the network to large images,
# since otherwise the resolution would be limited by the GPU memory." - Ronneberger et al 2015, U-Net paper

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from tiler import Tiler, Merger

# Loading image
# Photo by Christian Holzinger on Unsplash: https://unsplash.com/photos/CUY_YHhCFl4
image = np.array(Image.open("example_image.jpg"))  # 1280x1920x3

# Overlap tile strategy assumes we only use the small, non-overlapping, middle part of a tile.
# Assuming we want to use centre 64x64 square, we should specify
# The overlap should be 0.5, 64 or explicitly (64, 64, 0)
tiler = Tiler(
    data_shape=image.shape,
    tile_shape=(128, 128, 3),
    overlap=(64, 64, 0),
    channel_dimension=2,
)

# Calculate and apply extra padding, as well as adjust tiling parameters
new_shape, padding = tiler.calculate_padding()
tiler.recalculate(data_shape=new_shape)
padded_image = np.pad(image, padding, mode="reflect")

# Specifying merging parameters
# You can define overlap-tile window explicitly, i.e.
# >>> window = np.zeros((128, 128, 3))
# >>> window[32:-32, 32:-32, :] = 1
# >>> merger = Merger(tiler=tiler, window=window)
# or you can use window="overlap-tile"
# it will automatically calculate such window based on tiler.overlap and applied padding
merger = Merger(tiler=tiler, window="overlap-tile")

# Let's define a function that will be applied to each tile
# For this example, let's black out the sides that should be "cropped" by window function
# as a way to confirm that only the middle parts are being merged
def process(patch: np.ndarray) -> np.ndarray:
    patch[:32, :, :] = 0
    patch[-32:, :, :] = 0
    patch[:, :32, :] = 0
    patch[:, -32:, :] = 0
    return patch


# Iterate through all the tile and apply the processing function
# as well as add them back to the merger
for tile_id, tile in tiler(padded_image, progress_bar=True):
    processed_tile = process(tile)
    merger.add(tile_id, processed_tile)

# Merge processed tiles
final_image = merger.merge(extra_padding=padding, dtype=image.dtype)

print(f"Sanity check: {np.all(image == final_image)}")

# Show the final merged image, weights and number of times each pixel was seen in tiles
fig, ax = plt.subplots(3, 2)
ax[0, 0].set_title("Original image")
ax[0, 0].imshow(image)
ax[0, 1].set_title("Final merged image")
ax[0, 1].imshow(final_image)

ax[1, 0].set_title("Padded image")
ax[1, 0].imshow(padded_image)
ax[1, 1].set_title("Overlap-tile window")
ax[1, 1].imshow(merger.window)

ax[2, 0].set_title("Weights sum")
ax[2, 0].imshow(merger.weights_sum[:, :, 0], vmin=0, vmax=merger.weights_sum.max())
ax[2, 1].set_title("Pixel visits")
ax[2, 1].imshow(merger.data_visits[:, :, 0], vmin=0, vmax=merger.data_visits.max())
plt.show()
