import unittest
import numpy as np
from tiler import Tiler, Merger
from contextlib import redirect_stderr
import os


class TestMergingCommon(unittest.TestCase):

    def setUp(self) -> None:
        self.data = np.arange(0, 100)

    def test_init(self):
        tiler = Tiler(data_shape=self.data.shape,
                      tile_shape=(10, ))

        # logits test
        with self.assertRaises(ValueError):
            Merger(tiler=tiler,
                   logits=-1)
        with self.assertRaises(ValueError):
            Merger(tiler=tiler,
                  logits='unsupported_type')

        merger = Merger(tiler=tiler)
        np.testing.assert_equal(merger.data.shape, self.data.shape)

        merger2 = Merger(tiler=tiler,
                         logits=99)
        np.testing.assert_equal(merger2.data.shape, (99, ) + self.data.shape)

    def test_add(self):
        tiler = Tiler(data_shape=self.data.shape,
                      tile_shape=(10, ))
        tiler2 = Tiler(data_shape=self.data.shape,
                       tile_shape=(12, ),
                       mode='irregular')
        tiler3 = Tiler(data_shape=(3,) + self.data.shape,
                       tile_shape=(3, 10, ), channel_dimension=0)

        merger = Merger(tiler)
        merger_logits = Merger(tiler, logits=3)
        merger_irregular = Merger(tiler2)
        merger_channel_dim = Merger(tiler3)

        tile = tiler.get_tile(self.data, 0)
        tile_logits = np.vstack((tile, tile, tile))
        tile_irregular = tiler2.get_tile(self.data, len(tiler2) - 1)

        # Wrong tile id cases
        with self.assertRaises(IndexError):
            merger.add(-1, np.ones((10, )))
        with self.assertRaises(IndexError):
            merger.add(len(tiler), np.ones((10, )))

        # Usual mergers expect tile_shape == data_shape
        with self.assertRaises(ValueError):
            merger.add(0, np.ones((3, 10, )))
        merger.add(0, tile)
        np.testing.assert_equal(merger.merge()[:10], tile)

        # Logits merger expects an extra dimension in front for logits
        with self.assertRaises(ValueError):
            merger_logits.add(0, np.ones((10, )))
        merger_logits.add(0, tile_logits)
        np.testing.assert_equal(merger_logits.merge()[:, :10], tile_logits)
        np.testing.assert_equal(merger_logits.merge(argmax=True)[:10], np.zeros((10, )))

        # Irregular merger expects all(data_shape <= tile_shape)
        with self.assertRaises(ValueError):
            merger_irregular.add(0, np.ones((13, )))
        merger_irregular.add(len(tiler2) - 1, tile_irregular)
        np.testing.assert_equal(merger_irregular.merge()[-len(tile_irregular):], tile_irregular)

        # Channel dimension merger
        with self.assertRaises(ValueError):
            merger_channel_dim.add(0, np.ones((10, )))
        merger_channel_dim.add(0, tile_logits)
        np.testing.assert_equal(merger_channel_dim.merge()[:, :10], tile_logits)

        # gotta get that 100% coverage
        # this should just print a warning
        # let's suppress it to avoid confusion
        with open(os.devnull, "w") as null:
            with redirect_stderr(null):
                merger.set_window('boxcar')

    def test_batch_add(self):
        tiler = Tiler(data_shape=self.data.shape,
                      tile_shape=(10,))
        merger = Merger(tiler)

        batch1 = [x for _, x in tiler(self.data, False, batch_size=1)]
        np.testing.assert_equal(len(batch1), 10)
        np.testing.assert_equal(batch1[0].shape, (1, 10, ))
        for i, b in enumerate(batch1):
            merger.add_batch(i, 1, b)
        np.testing.assert_equal(merger.merge(), self.data)
        merger.reset()

        batch10 = [x for _, x in tiler(self.data, False, batch_size=10)]
        for i, b in enumerate(batch10):
            merger.add_batch(i, 10, b)
        np.testing.assert_equal(merger.merge(), self.data)
        merger.reset()

        batch8 = [x for _, x in tiler(self.data, False, batch_size=8)]
        np.testing.assert_equal(len(batch8), 2)
        np.testing.assert_equal(batch8[0].shape, (8, 10, ))
        np.testing.assert_equal(batch8[1].shape, (2, 10, ))
        for i, b in enumerate(batch8):
            merger.add_batch(i, 8, b)
        np.testing.assert_equal(merger.merge(), self.data)
        merger.reset()

        batch8_drop = [x for _, x in tiler(self.data, False, batch_size=8, drop_last=True)]
        np.testing.assert_equal(len(batch8_drop), 1)
        np.testing.assert_equal(batch8_drop[0].shape, (8, 10, ))
        for i, b in enumerate(batch8_drop):
            merger.add_batch(i, 8, b)
        np.testing.assert_equal(merger.merge()[:80], self.data[:80])
        np.testing.assert_equal(merger.merge()[80:], np.zeros((20,)))

        with self.assertRaises(IndexError):
            merger.add_batch(-1, 10, batch10[0])

        with self.assertRaises(IndexError):
            merger.add_batch(10, 10, batch10[9])

    def test_generate_window(self):
        tiler = Tiler(data_shape=self.data.shape,
                      tile_shape=(10,))

        with self.assertRaises(ValueError):
            Merger(tiler=tiler, window='unsupported_window')

        with self.assertRaises(ValueError):
            Merger(tiler=tiler, window=np.zeros((10, 10)))

        with self.assertRaises(ValueError):
            Merger(tiler=tiler, window=10)

        window = np.zeros((10, ))
        window[1:10] = 1
        merger = Merger(tiler=tiler, window=window)
        for t_id, t in tiler(self.data):
            merger.add(t_id, t)
        np.testing.assert_equal(merger.merge(),
                                [i if i % 10 else 0 for i in range(100)])

    def test_merge(self):

        # Test padding
        tiler = Tiler(data_shape=self.data.shape,
                      tile_shape=(12,))
        merger = Merger(tiler)
        for t_id, t in tiler(self.data):
            merger.add(t_id, t)

        np.testing.assert_equal(merger.merge(unpad=True), self.data)
        np.testing.assert_equal(merger.merge(unpad=False), np.hstack((self.data, [0, 0, 0, 0, 0, 0, 0, 0])))

        # Test argmax
        merger = Merger(tiler, logits=3)
        for t_id, t in tiler(self.data):
            merger.add(t_id, np.vstack((t, t / 2, t / 3)))

        np.testing.assert_equal(merger.merge(unpad=True, argmax=True), np.zeros((100, )))
        np.testing.assert_equal(merger.merge(unpad=True, argmax=False),
                                np.vstack((self.data, self.data / 2, self.data / 3)))

        np.testing.assert_equal(merger.merge(unpad=False, argmax=True), np.zeros((108, )))
        np.testing.assert_equal(merger.merge(unpad=False, argmax=False),
                                np.vstack((np.hstack((self.data, [0, 0, 0, 0, 0, 0, 0, 0])),
                                           np.hstack((self.data, [0, 0, 0, 0, 0, 0, 0, 0])) / 2,
                                           np.hstack((self.data, [0, 0, 0, 0, 0, 0, 0, 0])) / 3)))
