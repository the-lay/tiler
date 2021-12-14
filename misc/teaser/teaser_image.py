from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tiler import Tiler, Merger


#### Generate images
# Original image by Marco Bianchetti on Unsplash
# https://unsplash.com/photos/8blA_V0MI9I
image = np.array(Image.open('original_image.jpg'))

# Calculate tile shape to have exactly 3x3 tiles with 0.4 overlap
full_image_shape = np.array(image.shape)
full_image_shape[:2] = full_image_shape[:2] * 1.4
tile_shape = full_image_shape[:2] // 3
tile_shape = tuple(tile_shape) + (3, )

# Tile original image and merge it back
tiler = Tiler(image.shape, tile_shape, overlap=0.4, channel_dimension=2, mode='reflect')
tiles = np.array([tile for _, tile in tiler(image)])
merger = Merger(tiler)
merger.add_batch(0, len(tiler), tiles)
merged_image = merger.merge(dtype=tiles.dtype)


#### Plot images
fig = plt.figure(constrained_layout=True, figsize=(16, 4))
gs = fig.add_gridspec(3, 11)

# Original image
org_image = fig.add_subplot(gs[:, :3])
org_image.set_xticks([])
org_image.set_yticks([])
org_image.axis('off')
org_image.imshow(image)

# Tiler arrow
tiler_arrow = fig.add_subplot(gs[1, 3])
tiler_arrow.set_xlim((0, 100))
tiler_arrow.set_ylim((0, 100))
tiler_arrow.set_xticks([])
tiler_arrow.set_yticks([])
tiler_arrow.axis('off')
ta_text = tiler_arrow.text(50, 50, 'tiler.Tiler')
ta_text.set_ha('center')
ta_text2 = tiler_arrow.text(50, 36, '40% overlap')
ta_text2.set_ha('center')
tiler_arrow.arrow(0, 48, 95, 0, head_width=5, head_length=5, fc='k', ec='k')

# Tiles
tile_00 = fig.add_subplot(gs[0, 4])
tile_00.imshow(tiles[0])
tile_00.set_xticks([])
tile_00.set_yticks([])
tile_00.axis('off')

tile_01 = fig.add_subplot(gs[0, 5])
tile_01.imshow(tiles[1])
tile_01.set_xticks([])
tile_01.set_yticks([])
tile_01.axis('off')

tile_02 = fig.add_subplot(gs[0, 6])
tile_02.imshow(tiles[2])
tile_02.set_xticks([])
tile_02.set_yticks([])
tile_02.axis('off')

tile_10 = fig.add_subplot(gs[1, 4])
tile_10.imshow(tiles[3])
tile_10.set_xticks([])
tile_10.set_yticks([])
tile_10.axis('off')

tile_11 = fig.add_subplot(gs[1, 5])
tile_11.imshow(tiles[4])
tile_11.set_xticks([])
tile_11.set_yticks([])
tile_11.axis('off')

tile_12 = fig.add_subplot(gs[1, 6])
tile_12.imshow(tiles[5])
tile_12.set_xticks([])
tile_12.set_yticks([])
tile_12.axis('off')

tile_20 = fig.add_subplot(gs[2, 4])
tile_20.imshow(tiles[6])
tile_20.set_xticks([])
tile_20.set_yticks([])
tile_20.axis('off')

tile_21 = fig.add_subplot(gs[2, 5])
tile_21.imshow(tiles[7])
tile_21.set_xticks([])
tile_21.set_yticks([])
tile_21.axis('off')

tile_22 = fig.add_subplot(gs[2, 6])
tile_22.imshow(tiles[8])
tile_22.set_xticks([])
tile_22.set_yticks([])
tile_22.axis('off')

# Merger arrow
merger_arrow = fig.add_subplot(gs[1, 7])
merger_arrow.set_xlim((0, 100))
merger_arrow.set_ylim((0, 100))
merger_arrow.set_xticks([])
merger_arrow.set_yticks([])
merger_arrow.axis('off')
ma_text = merger_arrow.text(50, 50, 'tiler.Merger')
ma_text.set_ha('center')
ma_text2 = merger_arrow.text(50, 36, 'w/ window function')
ma_text2.set_ha('center')
merger_arrow.arrow(0, 48, 95, 0, head_width=5, head_length=5, fc='k', ec='k')

# Merged image
merged_img = fig.add_subplot(gs[:, 8:])
merged_img.set_xticks([])
merged_img.set_yticks([])
merged_img.axis('off')
merged_img.imshow(merged_image)

# fig.show()
plt.savefig('tiler_teaser.png')
