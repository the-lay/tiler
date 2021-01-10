import numpy as np
import sys
from typing import Union, Tuple, List
from scipy.signal.windows import get_window
from tiler import Tiler

class Merger:

    __WINDOWS = ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett',
                 'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann']

    def __init__(self,
                 tiler: Tiler,
                 window: str = 'boxcar',
                 logits: int = 0
                 ):
        """
        Merger precomputes everything for merging together tiles created by given Tiler.
        The merging is done according to the selected mode.

        :param tiler: Tiler
            Tiler with which the tiles are created.

        :param window: str
            Specify which window function to use for merging (tapering) of tiles.
            We use scipy to generate windows (scipy.signal.get_window()).

            One of the following string values:
                'boxcar' (default)
                    Boxcar window: the weight of each is tile array element is 1.
                    Also known as rectangular window or Dirichlet window (and equivalent to no window at all).
                'triang'
                    Triangular window.
                'blackman'
                    Blackman window.
                'hamming'
                    Hamming window.
                'hann'
                    Hann window.
                'bartlett'
                    Bartlett window.
                'flattop'
                    Flat top window.
                'parzen'
                    Parzen window.
                'bohman'
                    Bohman window.
                'blackmanharris'
                    Minimum 4-term Blackman-Harris window.
                'nuttall'
                    Minimum 4-term Blackman-Harris window according to Nuttall
                'barthann'
                    Bartlett-Hann window.

        :param logits: int
            If logits > 0, adds an extra dimension in front for logits in the data array.
            Default is 0.

        """

        self.tiler = tiler

        # Logits support
        if not isinstance(logits, int) or logits < 0:
            raise ValueError(f'Logits must be an integer 0 or a positive number ({logits}).')
        self.logits = int(logits)

        # Generate data and normalization arrays
        self.data = self.normalization = self.data_weights = None
        self.reset()

        # TODO generate window depending on tile border type
        # 1d = 3 types of tiles: 2 corners and middle
        # 2d = 9 types of tiles: 4 corners, 4 tiles with 1 edge and middle
        # 3d = 25 types of tiles: 8 corners, 12 tiles with 2 edges, 6 tiles with one edge and middle
        # corners: 2^ndim
        # tiles: 2*ndim*nedges

        # Generate window function
        self.window = None
        self.set_window(window)

    def _generate_window(self, window: str, shape: Union[Tuple, List]) -> np.ndarray:
        """
        Generate n-dimensional window according to the given shape.
        Adapted from: https://stackoverflow.com/a/53588640/1668421

        :param window: str
            Specify window function.
            We use scipy to generate windows (scipy.signal.get_window()).

            One of the following string values:
                'boxcar' (default)
                    Boxcar window: the weight of each is tile array element is 1.
                    Also known as rectangular window or Dirichlet window (and equivalent to no window at all).
                'triang'
                    Triangular window.
                'blackman'
                    Blackman window.
                'hamming'
                    Hamming window.
                'hann'
                    Hann window.
                'bartlett'
                    Bartlett window.
                'flattop'
                    Flat top window.
                'parzen'
                    Parzen window.
                'bohman'
                    Bohman window.
                'blackmanharris'
                    Minimum 4-term Blackman-Harris window.
                'nuttall'
                    Minimum 4-term Blackman-Harris window according to Nuttall
                'barthann'
                    Bartlett-Hann window.

        :param shape: tuple or list
            Shape of the requested window.

        :return: np.ndarray
        """

        w = np.ones(shape)
        for axis, length in enumerate(shape):
            if self.tiler.channel_dimension == axis:
                # channel dimension should have weight of 1 everywhere
                win = get_window('boxcar', length)
            else:
                win = get_window(window, length)

            for i in range(len(shape)):
                if i == axis:
                    continue
                else:
                    win = np.stack([win] * shape[i], axis=i)

            w *= win

        return w

    def set_window(self, window: str) -> None:
        """
        Generates windows for each tile depending on the given window function.

        :param window: str
            Specify which window function to use for merging (tapering) of tiles.
            We use scipy to generate windows (scipy.signal.get_window()).

            One of the following string values:
                'boxcar' (default)
                    Boxcar window: the weight of each is tile array element is 1.
                    Also known as rectangular window or Dirichlet window (and equivalent to no window at all).
                'triang'
                    Triangular window.
                'blackman'
                    Blackman window.
                'hamming'
                    Hamming window.
                'hann'
                    Hann window.
                'bartlett'
                    Bartlett window.
                'flattop'
                    Flat top window.
                'parzen'
                    Parzen window.
                'bohman'
                    Bohman window.
                'blackmanharris'
                    Minimum 4-term Blackman-Harris window.
                'nuttall'
                    Minimum 4-term Blackman-Harris window according to Nuttall
                'barthann'
                    Bartlett-Hann window.

        :return: None
        """

        if window not in self.__WINDOWS:
            raise ValueError('Unsupported window, please check docs')

        # Warn user that changing window type after some elements were already visited is a bad idea.
        if np.count_nonzero(self.normalization):
            print('Warning: you are changing a window type after some elements '
                  ' were already processed and that might lead to an unpredicted behavior.', file=sys.stderr)

        # Generate and set the window
        self.window = self._generate_window(window, self.tiler.tile_shape)

        # Border calculations
        # TODO border

        # TODO
        # TODO depending on the mode, generate windows
        # # Tile border type calculations
        # # 2D: Corner elements will be 0, edge elements will be 1, center elements will be 2
        # # TODO
        # self._tile_border_types = np.zeros(self._indexing_shape, dtype=np.int32)
        # for i in range(self._n_dim):
        #     sl = [slice(None, None, None) for _ in range(self._n_dim)]
        #     sl[i] = slice(1, -1)
        #     self._tile_border_types[tuple(sl)] += 1
        #
        # # Processed data will hold the data
        # self.merged_image = np.zeros_like(self.image_shape)
        # self.elements_visited = np.zeros_like(self.image_shape, dtype=np.uint)
        # TODO TODO TODO
        # print('adfgadf')

    def reset(self) -> None:
        """
        Reset data and normalization buffers.
        Should be done after finishing merging full tile set and before starting processing the next tile set.

        :return: None
        """

        padded_data_shape = self.tiler._new_shape

        # Image holds sum of all processed tiles multiplied by the window
        if self.logits:
            self.data = np.zeros([self.logits] + padded_data_shape)
        else:
            self.data = np.zeros(padded_data_shape)

        # Normalization array holds the number of times each element was visited
        self.normalization = np.zeros(padded_data_shape, dtype=np.uint32)

        # Total data window (weight) coefficients
        self.data_weights = np.ones(padded_data_shape)

    def add(self, tile_id: int, data: np.ndarray) -> None:
        """
        Adds processed tile back into Merger as a tile number tile_id.

        :param tile_id: int
            Specify which tile it is.

        :param data: np.ndarray
            Tile data.

        :return: None
        """

        if tile_id < 0 or tile_id >= len(self.tiler):
            raise IndexError(f'Out of bounds, there is no tile {tile_id}. '
                             f'There are {len(self.tiler)} tiles, starting from index 0.')

        data_shape = np.array(data.shape)
        expected_tile_shape = ((self.logits, ) + tuple(self.tiler.tile_shape)) if self.logits > 0 else tuple(self.tiler.tile_shape)

        if self.tiler.mode != 'irregular':
            if not np.all(np.equal(data_shape, expected_tile_shape)):
                raise ValueError(f'Passed data shape ({data_shape}) '
                                 f'does not fit expected tile shape ({expected_tile_shape}).')
        else:
            if not np.all(np.less_equal(data_shape, expected_tile_shape)):
                raise ValueError(f'Passed data shape ({data_shape}) '
                                 f'must be less or equal than tile shape ({expected_tile_shape}).')

        # Select coordinates for data
        shape_diff = expected_tile_shape - data_shape
        a, b = self.tiler.get_tile_bbox_position(tile_id, with_channel_dim=True)

        sl = [slice(x, y - shape_diff[i]) for i, (x, y) in enumerate(zip(a, b))]
        win_sl = [slice(None, -diff) if (diff > 0) else slice(None, None) for diff in shape_diff]

        if self.logits > 0:
            self.data[tuple([slice(None, None, None)] + sl)] += data
            self.data_weights[tuple(sl)] *= self.window[tuple(win_sl[1:])]
        else:
            self.data[tuple(sl)] += data
            self.data_weights[tuple(sl)] *= self.window[tuple(win_sl)]
        self.normalization[tuple(sl)] += 1

        # Add processed tile data
        # Data array holds data with window applied
        # Data weights holds final weight applied to each element
        # Normalization holds number of times each element was modified
        # self.data[tuple(sl)] += (data * self.window[tuple(win_sl)])
        # self.data_weights[tuple(sl)] *= self.window[tuple(win_sl)]
        # self.normalization[tuple(sl)] += 1

    def merge(self, unpad: bool = True, normalize: bool = True, argmax: bool = False) -> np.ndarray:
        """
        Returns final merged data array obtained from added tiles.

        :param unpad: bool
            If unpad is True, removes padded elements.
            Default is True.

        :param normalize: bool
            If normalize is True, divides elements by the number of times they were visited (self.normalization).
            Default is True.

        :param argmax: bool
            If argmax is True, the first dimension will be argmaxed.
            Default is False.

        :return: np.ndarray
            Final merged data array obtained from added tiles.
        """

        data = self.data * self.data_weights

        if normalize:
            # context to remove division by zero warnings
            with np.errstate(divide='ignore', invalid='ignore'):
                data = data / self.normalization

        if argmax:
            data = np.argmax(data, 0)

        if unpad:
            sl = [slice(None, self.tiler.image_shape[i]) for i in range(len(self.tiler.image_shape))]
            data = data[tuple(sl)]

        return data
