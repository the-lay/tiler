# Again, this is almost verbatim from scipy's window tests
import unittest
import numpy as np

from tiler import _windows
from tiler import Merger


class TestGetWindow(unittest.TestCase):
    def test_boxcar(self):
        w = _windows.get_window("boxcar", 12)
        np.testing.assert_array_equal(w, np.ones_like(w))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            _windows.get_window("boxcar", "wrong")
        with self.assertRaises(ValueError):
            _windows.get_window("no_such_window", 4)

    def test_window_guards(self):
        windows = Merger.SUPPORTED_WINDOWS
        windows.remove("overlap-tile")
        for window in windows:
            with self.assertRaises(ValueError):
                _windows.get_window(window, -1)
            with self.assertRaises(ValueError):
                _windows.get_window(window, 1.3)
            with self.assertRaises(ValueError):
                _windows.get_window(window, "hello")

            # triggered len guards
            w = _windows.get_window(window, 1)
            np.testing.assert_equal(w, 1)


class TestBoxcar(unittest.TestCase):
    def test_basic(self):
        np.testing.assert_allclose(_windows._boxcar(6), [1, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(_windows._boxcar(7), [1, 1, 1, 1, 1, 1, 1])


class TestTriang(unittest.TestCase):
    def test_basic(self):
        np.testing.assert_allclose(
            _windows._triang(6), [1 / 6, 1 / 2, 5 / 6, 5 / 6, 1 / 2, 1 / 6]
        )
        np.testing.assert_allclose(
            _windows._triang(7), [1 / 4, 1 / 2, 3 / 4, 1, 3 / 4, 1 / 2, 1 / 4]
        )


class TestParzen(unittest.TestCase):
    def test_basic(self):
        np.testing.assert_allclose(
            _windows._parzen(6),
            [
                0.009259259259259254,
                0.25,
                0.8611111111111112,
                0.8611111111111112,
                0.25,
                0.009259259259259254,
            ],
        )
        np.testing.assert_allclose(
            _windows._parzen(7),
            [
                0.00583090379008747,
                0.1574344023323616,
                0.6501457725947521,
                1.0,
                0.6501457725947521,
                0.1574344023323616,
                0.00583090379008747,
            ],
        )


class TestBohman(unittest.TestCase):
    def test_basic(self):
        np.testing.assert_allclose(
            _windows._bohman(6),
            [
                0,
                0.1791238937062839,
                0.8343114522576858,
                0.8343114522576858,
                0.1791238937062838,
                0,
            ],
        )
        np.testing.assert_allclose(
            _windows._bohman(7),
            [
                0,
                0.1089977810442293,
                0.6089977810442293,
                1.0,
                0.6089977810442295,
                0.1089977810442293,
                0,
            ],
        )


class TestBlackman(unittest.TestCase):
    def test_basic(self):
        np.testing.assert_allclose(
            _windows._blackman(6),
            [
                0,
                0.2007701432625305,
                0.8492298567374694,
                0.8492298567374694,
                0.2007701432625305,
                0,
            ],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            _windows._blackman(7), [0, 0.13, 0.63, 1.0, 0.63, 0.13, 0], atol=1e-14
        )


class TestNuttall(unittest.TestCase):
    def test_basic(self):
        np.testing.assert_allclose(
            _windows._nuttall(6),
            [
                0.0003628,
                0.1105152530498718,
                0.7982580969501282,
                0.7982580969501283,
                0.1105152530498719,
                0.0003628,
            ],
        )
        np.testing.assert_allclose(
            _windows._nuttall(7),
            [0.0003628, 0.0613345, 0.5292298, 1.0, 0.5292298, 0.0613345, 0.0003628],
        )


class TestBlackmanHarris(unittest.TestCase):
    def test_basic(self):
        np.testing.assert_allclose(
            _windows._blackmanharris(6),
            [
                6.0e-05,
                0.1030114893456638,
                0.7938335106543362,
                0.7938335106543364,
                0.1030114893456638,
                6.0e-05,
            ],
        )
        np.testing.assert_allclose(
            _windows._blackmanharris(7),
            [6.0e-05, 0.055645, 0.520575, 1.0, 0.520575, 0.055645, 6.0e-05],
        )


class TestBartlett(unittest.TestCase):
    def test_basic(self):
        np.testing.assert_allclose(_windows._bartlett(6), [0, 0.4, 0.8, 0.8, 0.4, 0])
        np.testing.assert_allclose(
            _windows._bartlett(7), [0, 1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3, 0]
        )


class TestHann(unittest.TestCase):
    def test_basic(self):
        np.testing.assert_allclose(
            _windows._hann(6),
            [
                0,
                0.3454915028125263,
                0.9045084971874737,
                0.9045084971874737,
                0.3454915028125263,
                0,
            ],
        )
        np.testing.assert_allclose(
            _windows._hann(7), [0, 0.25, 0.75, 1.0, 0.75, 0.25, 0]
        )


class TestBartHann(unittest.TestCase):
    def test_basic(self):
        np.testing.assert_allclose(
            _windows._barthann(6),
            [
                0,
                0.35857354213752,
                0.8794264578624801,
                0.8794264578624801,
                0.3585735421375199,
                0,
            ],
        )
        np.testing.assert_allclose(
            _windows._barthann(7), [0, 0.27, 0.73, 1.0, 0.73, 0.27, 0]
        )


class TestHamming(unittest.TestCase):
    def test_basic(self):
        np.testing.assert_allclose(
            _windows._hamming(6),
            [
                0.08,
                0.3978521825875242,
                0.9121478174124757,
                0.9121478174124757,
                0.3978521825875242,
                0.08,
            ],
        )
        np.testing.assert_allclose(
            _windows._hamming(7), [0.08, 0.31, 0.77, 1.0, 0.77, 0.31, 0.08]
        )
