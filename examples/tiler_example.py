import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tiler import Tiler, Merger
plt.ioff()

# # Create a dummy data
# data = np.ones((256, 256))  # 256x256
# data *= np.vstack(np.arange(256) * 256)

# Load an data
image = np.array(Image.open('/home/ilja/Pictures/profile.jpg'))  # 1350x1080x3

# Create Tiler object that will handle tiling
# Mandatory arguments: image_shape, tile_shape
# Optional arguments: mode, channel_dimension, overlap
tiler = Tiler(image_shape=image.shape, tile_shape=(200, 200, 3), mode='irregular', channel_dimension=2, overlap=0.1)

# You can now call Tiler object with data that you wish to be split into tiles
# Tiler returns a generator with all tiles
for tile_id, tile in tiler(image):
    print(f'This is tile {tile_id} out of {len(tiler)} tiles.')
    # do anything with the tile here

# You can also get individual tiles
tile_3 = tiler.get_tile(image, 3)

# You can also use Merger to handle merging tiles back into one array
# Mandatory arguments: tiler
# Optional arguments: mode
merger = Merger(tiler=tiler, window='hamming')
for tile_id, tile in tiler(image):
    # processed_tile = some_processing(tile)
    processed_tile = (tile * 0.9)
    merger.add(tile_id, processed_tile)
processed_image = merger.merge(unpad=True, normalize=True)
fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
ax[0].imshow(processed_image / np.iinfo(image.dtype).max)
ax[1].imshow(merger.data_weights, cmap='jet')
ax[2].imshow(merger.normalization, cmap='jet')
plt.show()

# # Example: plot all tiles in one figure
# fig, ax = plt.subplots(*tiler.get_mosaic_shape(), tight_layout=True)
# for tile_i, tile in tiler(data):
#     x, y = tiler.get_tile_mosaic_position(tile_i)
#     ax[x, y].imshow(tile, vmin=data.min(), vmax=data.max())
# plt.show()

# Example: tile the data with tiler, process each tile and merge back into full data



