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
image = np.array(Image.open('example_image.jpg'))  # 1280x1920x3

# Padding image
# Overlap tile strategy assumes we only use the small, non-overlapping, middle part of tile.
# Assuming we want tiles of size 128x128 and we want to only use middle 64x64,
# we should pad the image by 32 from each side, using reflect mode
padded_image = np.pad(image, ((32, 32), (32, 32), (0, 0)), mode='reflect')

# Specifying tiling parameters
# The overlap should be 0.5, 64 or explicitly (64, 64, 0)
tiler = Tiler(data_shape=padded_image.shape, tile_shape=(128, 128, 3),
              overlap=(64, 64, 0), channel_dimension=2)

# Specifying merging parameters
# You can define overlap-tile window explicitly, i.e.
# window = np.zeros((128, 128, 3))
# window[32:-32, 32:-32, :] = 1
# merger = Merger(tiler=tiler, window=window)
# or you can use overlap-tile window which will do that automatically based on tiler.overlap
merger = Merger(tiler=tiler, window='overlap-tile')

# Let's define a function that will be applied to each tile
def process(patch: np.ndarray, sanity_check: bool = True) -> np.ndarray:

    # One example can be a sanity check
    # Make the parts that should be removed black
    # There should not appear any black spots in the final merged image
    if sanity_check:
        patch[:32, :, :] = 0
        patch[-32:, :, :] = 0
        patch[:, :32, :] = 0
        patch[:, -32:, :] = 0
        return patch

    # Another example can be to just modify the whole patch
    # Using PIL, we adjust the color balance
    enhancer = ImageEnhance.Color(Image.fromarray(patch))
    return np.array(enhancer.enhance(5.0))

# Iterate through all the tile and apply the processing function
# as well as add them back to the merger
for tile_id, tile in tiler(padded_image):
    processed_tile = process(tile)
    merger.add(tile_id, processed_tile)

# Merger.merge() returns unpadded from tiler image, but we still need to unpad line#21
final_image = merger.merge().astype(np.uint8)
final_unpadded_image = final_image[32:-32, 32:-32, :]

# Show the final merged image, weights and number of times each pixel was seen in tiles
fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
ax[0, 0].set_title('Original image')
ax[0, 0].imshow(image)
ax[0, 1].set_title('Final unpadded image')
ax[0, 1].imshow(final_unpadded_image)

ax[1, 0].set_title('Padded image')
ax[1, 0].imshow(padded_image)
ax[1, 1].set_title('Merged image')
ax[1, 1].imshow(final_image)

ax[2, 0].set_title('Weights sum')
ax[2, 0].imshow(merger.weights_sum[:, :, 0], vmin=0, vmax=merger.weights_sum.max())
ax[2, 1].set_title('Pixel visits')
ax[2, 1].imshow(merger.data_visits[:, :, 0], vmin=0, vmax=merger.data_visits.max())
plt.show()
