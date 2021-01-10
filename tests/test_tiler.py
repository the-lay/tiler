import unittest
import numpy as np
from tiler import Tiler

class TestTilingCommon(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(ValueError):
            Tiler(image_shape=(-10, -30), tile_shape=(10, 10))

        with self.assertRaises(ValueError):
            Tiler(image_shape=(100, 300), tile_shape=(-10, -10))

        with self.assertRaises(ValueError):
            Tiler(image_shape=(300, 300), tile_shape=(10, 10, 10))

        with self.assertRaises(ValueError):
            Tiler(image_shape=(300, 300), tile_shape=(10, 10),
                  mode='unsupported_mode')

        with self.assertRaises(ValueError):
            Tiler(image_shape=(300, 300), tile_shape=(10, 10),
                  channel_dimension=-1)

        with self.assertRaises(ValueError):
            Tiler(image_shape=(300, 300), tile_shape=(10, 10),
                  channel_dimension=10)

        with self.assertRaises(ValueError):
            Tiler(image_shape=(300, 300), tile_shape=(10, 10),
                  overlap=1337.0)

        with self.assertRaises(ValueError):
            Tiler(image_shape=(300, 300), tile_shape=(10, 10),
                  overlap=(15, 0))

        with self.assertRaises(ValueError):
            Tiler(image_shape=(300, 300), tile_shape=(10, 10),
                  overlap='unsupported_overlap')

    def test_repr(self):
        # gotta get that coverage
        tiler = Tiler(image_shape=(3, 300, 300), tile_shape=(3, 15, 300),
                      channel_dimension=0,
                      mode='irregular')

        expected_repr = '[3, 15, 300] tiler for data of shape [3, 300, 300]:' \
                        '\n\tNew shape: [3, 300, 300]' \
                        '\n\tOverlap: 0.0' \
                        '\n\tStep: [0, 15, 300]' \
                        '\n\tMode: irregular' \
                        '\n\tChannel dimension: 0'

        self.assertEqual(str(tiler), expected_repr)


class TestTiling1D(unittest.TestCase):

    def setUp(self):
        self.n_elements = 100
        self.data = np.arange(0, self.n_elements)

    def test_drop_mode(self):
        # Drop mode drops last uneven tile
        tile_size = 15
        tiler = Tiler(image_shape=self.data.shape,
                      tile_shape=(tile_size, ),
                      mode='drop')

        expected_split = [self.data[i:i + tile_size] for i in range(0, self.n_elements, tile_size)]
        expected_split = expected_split[:-1]

        calculated_split = [t for _, t in tiler(self.data)]

        self.assertEqual(len(tiler), len(expected_split))
        np.testing.assert_equal(expected_split, calculated_split)

    def test_irregular_mode(self):
        # Irregular mode returns last chunk even if it is not equal to the tile size
        tile_size = 15
        tiler = Tiler(image_shape=self.data.shape,
                      tile_shape=(tile_size, ),
                      mode='irregular')

        expected_split = [self.data[i:i + tile_size] for i in range(0, self.n_elements, tile_size)]

        calculated_split = [t for _, t in tiler(self.data)]

        self.assertEqual(len(tiler), len(expected_split))
        np.testing.assert_equal(expected_split, calculated_split)

    def test_constant_mode(self):
        # Constant mode pads the non-full tiles with constant_value
        tile_size = 15
        constant_value = -99
        tiler = Tiler(image_shape=self.data.shape,
                      tile_shape=(tile_size, ),
                      mode='constant', constant_value=constant_value)

        expected_split = [self.data[i:i + tile_size] for i in range(0, self.n_elements, tile_size)]
        expected_split[-1] = np.pad(expected_split[-1], (0, tile_size - len(expected_split[-1])),
                                    mode='constant', constant_values=constant_value)

        calculated_split = [t for _, t in tiler(self.data)]

        self.assertEqual(len(tiler), len(expected_split))
        np.testing.assert_equal(expected_split, calculated_split)

    def test_reflect_mode(self):
        # Reflect mode pads with reflected values along the axis
        tile_size = 15
        tiler = Tiler(image_shape=self.data.shape,
                      tile_shape=(tile_size, ),
                      mode='reflect')

        expected_split = [self.data[i:i + tile_size] for i in range(0, self.n_elements, tile_size)]
        expected_split[-1] = np.pad(expected_split[-1], (0, tile_size - len(expected_split[-1])), mode='reflect')

        calculated_split = [t for _, t in tiler(self.data)]

        self.assertEqual(len(tiler), len(expected_split))
        np.testing.assert_equal(expected_split, calculated_split)

    def test_edge_mode(self):
        # Edge mode pads with the edge values of data
        tile_size = 15
        tiler = Tiler(image_shape=self.data.shape,
                      tile_shape=(tile_size, ),
                      mode='edge')

        expected_split = [self.data[i:i + tile_size] for i in range(0, self.n_elements, tile_size)]
        expected_split[-1] = np.pad(expected_split[-1], (0, tile_size - len(expected_split[-1])), mode='edge')

        calculated_split = [t for _, t in tiler(self.data)]

        self.assertEqual(len(tiler), len(expected_split))
        np.testing.assert_equal(expected_split, calculated_split)

    def test_wrap_mode(self):
        # Wrap mode pads the tile with the wrap of the vector along the axis
        tile_size = 15
        tiler = Tiler(image_shape=self.data.shape,
                      tile_shape=(tile_size, ),
                      mode='wrap')

        expected_split = [self.data[i:i + tile_size] for i in range(0, self.n_elements, tile_size)]
        expected_split[-1] = np.pad(expected_split[-1], (0, tile_size - len(expected_split[-1])), mode='wrap')

        calculated_split = [t for _, t in tiler(self.data)]

        self.assertEqual(len(tiler), len(expected_split))
        np.testing.assert_equal(expected_split, calculated_split)

    def test_channel_dimensions(self):
        # If a channel dimension is specified, the tiler should return the data without splitting
        tile_size = 15
        data = np.vstack((self.data, self.data * 2, self.data * 3))
        tiler = Tiler(image_shape=data.shape,
                      tile_shape=(3, tile_size),
                      mode='irregular', channel_dimension=0)

        expected_split = [[data[0][i:i + tile_size],
                           data[1][i:i + tile_size],
                           data[2][i:i + tile_size]]
                          for i in range(0, self.n_elements, tile_size)]

        calculated_split = [t for _, t in tiler(data)]

        self.assertEqual(len(tiler), len(expected_split))
        np.testing.assert_equal(expected_split, calculated_split)

    def test_get_tile(self):
        tile_size = 10
        tiler = Tiler(image_shape=self.data.shape,
                      tile_shape=(tile_size, ))

        with self.assertRaises(IndexError):
            tiler.get_tile(self.data, len(tiler))
        with self.assertRaises(IndexError):
            tiler.get_tile(self.data, -1)

        tiles = tiler.view_in_tiles(self.data)
        np.testing.assert_equal([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], tiler.get_tile(None, 0, tiles))

        np.testing.assert_equal([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], tiler.get_tile(self.data, 0))

    def test_view_in_tiles(self):
        tiler = Tiler(image_shape=self.data.shape,
                      tile_shape=(10, ))
        with self.assertRaises(ValueError):
            tiler.view_in_tiles(self.data.reshape((10, 10)))

    def test_overlap(self):
        # Case 1
        # If overlap is an integer, the same overlap should be applied in each dimension
        tile_size = 10
        overlap = 5
        tiler = Tiler(image_shape=self.data.shape,
                      tile_shape=(tile_size, ), overlap=overlap)

        expected_split = [[i for i in range(j, j + tile_size)]
                          for j in range(0, self.n_elements - overlap, tile_size - overlap)]

        calculated_split = [t for _, t in tiler(self.data)]

        self.assertEqual(len(tiler), len(expected_split))
        np.testing.assert_equal(expected_split, calculated_split)

        # Case 2
        # If overlap is a float, compute the actual number
        tile_size = 10
        overlap = 0.5
        el_overlap = int(tile_size * overlap)
        tiler = Tiler(image_shape=self.data.shape,
                      tile_shape=(tile_size, ), overlap=overlap)

        expected_split = [[i for i in range(j, j + tile_size)]
                          for j in range(0, self.n_elements - el_overlap, tile_size - el_overlap)]

        calculated_split = [t for _, t in tiler(self.data)]

        self.assertEqual(len(tiler), len(expected_split))
        np.testing.assert_equal(expected_split, calculated_split)

        # Case 3
        # Overlap is provided as tuple or list
        # Let's try a slightly more complicated test case with a channel dimension
        tile_size = 10
        data = np.vstack((self.data, self.data * 2, self.data * 3))
        overlap = (0, 5, )
        tiler = Tiler(image_shape=data.shape,
                      tile_shape=(3, tile_size, ), overlap=overlap)

        expected_split = [[[i for i in range(j, j + tile_size)],
                           [i * 2 for i in range(j, j + tile_size)],
                           [i * 3 for i in range(j, j + tile_size)]]
                          for j in range(0, self.n_elements - overlap[1], tile_size - overlap[1])]

        calculated_split = [t for _, t in tiler(data)]

        self.assertEqual(len(tiler), len(expected_split))
        np.testing.assert_equal(expected_split, calculated_split)

    def test_tile_sample_shape(self):
        tile_size = 10
        tiler = Tiler(image_shape=self.data.shape,
                      tile_shape=(tile_size, ),
                      channel_dimension=None)
        tiler2 = Tiler(image_shape=(3, ) + self.data.shape,
                       tile_shape=(3, tile_size),
                       channel_dimension=0)

        with self.assertRaises(IndexError):
            tiler.get_tile_sample_shape(len(tiler))

        np.testing.assert_equal([tile_size], tiler.get_tile_sample_shape(len(tiler) - 1))
        np.testing.assert_equal([tile_size], tiler2.get_tile_sample_shape(len(tiler) - 1))

    def test_tile_mosaic_position(self):
        tile_size = 10
        tiler = Tiler(image_shape=self.data.shape, tile_shape=(tile_size, ))
        tiler2 = Tiler(image_shape=(3, ) + self.data.shape, tile_shape=(3, tile_size, ), channel_dimension=0)

        tile_id = 0
        np.testing.assert_equal([0], tiler.get_tile_mosaic_position(tile_id))
        np.testing.assert_equal([0], tiler.get_tile_mosaic_position(tile_id, with_channel_dim=True))
        np.testing.assert_equal([0], tiler2.get_tile_mosaic_position(tile_id))
        np.testing.assert_equal([0, 0], tiler2.get_tile_mosaic_position(tile_id, with_channel_dim=True))

        tile_id = len(tiler) - 1
        np.testing.assert_equal([9], tiler.get_tile_mosaic_position(tile_id))
        np.testing.assert_equal([9], tiler.get_tile_mosaic_position(tile_id, with_channel_dim=True))
        np.testing.assert_equal([9], tiler2.get_tile_mosaic_position(tile_id))
        np.testing.assert_equal([0, 9], tiler2.get_tile_mosaic_position(tile_id, with_channel_dim=True))

        with self.assertRaises(IndexError):
            tiler.get_tile_mosaic_position(-1)
        with self.assertRaises(IndexError):
            tiler.get_tile_mosaic_position(len(tiler))

    def test_mosaic_shape(self):
        tile_size = 10
        tiler = Tiler(image_shape=self.data.shape, tile_shape=(tile_size, ))
        tiler2 = Tiler(image_shape=(3, ) + self.data.shape, tile_shape=(3, tile_size, ), channel_dimension=0)
        tiler3 = Tiler(image_shape=(9, ) + self.data.shape, tile_shape=(3, tile_size, ), channel_dimension=0)
        tiler4 = Tiler(image_shape=(9, ) + self.data.shape, tile_shape=(3, tile_size, ))

        np.testing.assert_equal([10], tiler.get_mosaic_shape())
        np.testing.assert_equal([10], tiler.get_mosaic_shape(with_channel_dim=True))

        np.testing.assert_equal([10], tiler2.get_mosaic_shape())
        np.testing.assert_equal([1, 10], tiler2.get_mosaic_shape(with_channel_dim=True))

        np.testing.assert_equal([10], tiler3.get_mosaic_shape())
        np.testing.assert_equal([1, 10], tiler3.get_mosaic_shape(with_channel_dim=True))

        np.testing.assert_equal([3, 10], tiler4.get_mosaic_shape())
        np.testing.assert_equal([3, 10], tiler4.get_mosaic_shape(with_channel_dim=True))

    def test_tile_bbox_position(self):
        tile_size = 10
        tiler = Tiler(image_shape=self.data.shape, tile_shape=(tile_size, ))
        tiler2 = Tiler(image_shape=(3, ) + self.data.shape, tile_shape=(3, tile_size, ), channel_dimension=0)

        with self.assertRaises(IndexError):
            tiler.get_tile_bbox_position(-1)
        with self.assertRaises(IndexError):
            tiler.get_tile_bbox_position(len(tiler))

        tile_id = 0
        np.testing.assert_equal(([0], [10]), tiler.get_tile_bbox_position(tile_id))
        np.testing.assert_equal(([0], [10]), tiler.get_tile_bbox_position(tile_id, True))
        np.testing.assert_equal(([0], [10]), tiler2.get_tile_bbox_position(tile_id))
        np.testing.assert_equal(([0, 0], [3, 10]), tiler2.get_tile_bbox_position(tile_id, True))

        tile_id = len(tiler) - 1
        np.testing.assert_equal(([90], [100]), tiler.get_tile_bbox_position(tile_id))
        np.testing.assert_equal(([90], [100]), tiler.get_tile_bbox_position(tile_id, True))
        np.testing.assert_equal(([90], [100]), tiler2.get_tile_bbox_position(tile_id))
        np.testing.assert_equal(([0, 90], [3, 100]), tiler2.get_tile_bbox_position(tile_id, True))
