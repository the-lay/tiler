import numpy as np
import sys
from typing import Union, Tuple, List
from scipy.signal.windows import get_window
from tiler import Tiler


class Merger:

    SUPPORTED_WINDOWS = ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett',
                         'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann']
    r"""
    Supported windows:
    - 'boxcar' (default)  
        Boxcar window: the weight of each is tile element is 1.  
        Also known as rectangular window or Dirichlet window (and equivalent to no window at all).
    - 'triang'  
        Triangular window.
    - 'blackman'  
        Blackman window.
    - 'hamming'  
        Hamming window.
    - 'hann'  
        Hann window.
    - 'bartlett'  
        Bartlett window.
    - 'flattop'  
        Flat top window.
    - 'parzen'  
        Parzen window.
    - 'bohman'  
        Bohman window.
    - 'blackmanharris'  
        Minimum 4-term Blackman-Harris window.
    - 'nuttall'  
        Minimum 4-term Blackman-Harris window according to Nuttall
    - 'barthann'  
        Bartlett-Hann window.
    """

    def __init__(self,
                 tiler: Tiler,
                 window: Union[str, np.ndarray] = 'boxcar',
                 logits: int = 0):
        """Merger precomputes everything for merging together tiles created by given Tiler.

        TODO:
            - generate window depending on tile border type

        Args:
            tiler (Tiler): Tiler with which the tiles were originally created.

            window (str or np.ndarray): Specifies which window function to use for tile merging.
                Must be one of `Merger.SUPPORTED_WINDOWS` or a numpy array with the same size as the tile.
                Default is `boxcar`.

            logits (int): Specify whether to add logits dimensions in front of the data array. Default is `0`.
        """
        self.tiler = tiler

        # Logits support
        if not isinstance(logits, int) or logits < 0:
            raise ValueError(f'Logits must be an integer 0 or a positive number ({logits}).')
        self.logits = int(logits)

        # Generate data and normalization arrays
        self.data = self.data_visits = self.weights_sum = None
        self.reset()

        # for the future borders generation
        # 1d = 3 types of tiles: 2 corners and middle
        # 2d = 9 types of tiles: 4 corners, 4 tiles with 1 edge and middle
        # 3d = 25 types of tiles: 8 corners, 12 tiles with 2 edges, 6 tiles with one edge and middle
        # corners: 2^ndim
        # tiles: 2*ndim*nedges

        # Generate window function
        self.window = None
        self.set_window(window)

    def _generate_window(self, window: str, shape: Union[Tuple, List]) -> np.ndarray:
        """Generate n-dimensional window according to the given shape.
        Adapted from: https://stackoverflow.com/a/53588640/1668421
        We use scipy to generate windows (scipy.signal.get_window()).

        Args:
            window (str): Specifies window function. Must be one of `Merger.SUPPORTED_WINDOWS`.
            shape (tuple or list): Shape of the requested window.

        Returns:
            np.ndarray: n-dimensional window of the given shape and function
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

    def set_window(self, window: Union[str, np.ndarray]) -> None:
        """Sets window function depending on the given window function.

        Args:
            window (str or np.ndarray): Specifies which window function to use for tile merging.
                Must be one of `Merger.SUPPORTED_WINDOWS` or a numpy array with the same size as the tile.
                Default is `boxcar`.

        Returns:
            None
        """
        # Warn user that changing window type after some elements were already visited is a bad idea.
        if np.count_nonzero(self.data_visits):
            print('Warning: you are changing a window type after some elements '
                  ' were already processed and that might lead to an unpredicted behavior.', file=sys.stderr)

        # Generate or set a window function
        if isinstance(window, str):
            if window not in self.SUPPORTED_WINDOWS:
                raise ValueError('Unsupported window, please check docs')
            self.window = self._generate_window(window, self.tiler.tile_shape)
        elif isinstance(window, np.ndarray):
            if not np.array_equal(window.shape, self.tiler.tile_shape):
                raise ValueError(f'Window function must have the same shape as tile shape.')
            self.window = window
        else:
            raise ValueError(f'Unsupported type for window function ({type(window)}), expected str or np.ndarray.')

    def reset(self) -> None:
        """Reset data and normalization buffers.

        Should be done after finishing merging full tile set and before starting processing the next tile set.

        Returns:
            None
        """

        padded_data_shape = self.tiler._new_shape

        # Image holds sum of all processed tiles multiplied by the window
        if self.logits:
            self.data = np.zeros(np.hstack((self.logits, padded_data_shape)))
        else:
            self.data = np.zeros(padded_data_shape)

        # Normalization array holds the number of times each element was visited
        self.data_visits = np.zeros(padded_data_shape, dtype=np.uint32)

        # Total data window (weight) coefficients
        self.weights_sum = np.zeros(padded_data_shape)

    def add(self, tile_id: int, data: np.ndarray) -> None:
        """Adds `tile_id`-th tile into Merger.

        Args:
            tile_id (int): Specifies which tile it is.
            data (np.ndarray): Specifies tile data.

        Returns:
            None
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
            self.data[tuple([slice(None, None, None)] + sl)] += (data * self.window[tuple(win_sl[1:])])
            self.weights_sum[tuple(sl)] += self.window[tuple(win_sl[1:])]
        else:
            self.data[tuple(sl)] += (data * self.window[tuple(win_sl)])
            self.weights_sum[tuple(sl)] += self.window[tuple(win_sl)]
        self.data_visits[tuple(sl)] += 1

    def add_batch(self, batch_id: int, batch_size: int, data: np.ndarray) -> None:
        """Adds `batch_id`-th batch of `batch_size` tiles into Merger.

        Args:
            batch_id (int): Specifies batch number, must be >= 0.
            batch_size (int): Specifies batch size, must be >= 0.
            data (np.ndarray): Tile data array, must have shape `[batch, *tile_shape]

        Returns:
            None
        """

        # calculate total number of batches
        div, mod = np.divmod(len(self.tiler), batch_size)
        n_batches = (div + 1) if mod > 0 else div

        if batch_id < 0 or batch_id >= n_batches:
            raise IndexError(f'Out of bounds. There are {n_batches} batches of {batch_size}, starting from index 0.')

        # add each tile in a batch with computed tile_id
        for data_i, tile_i in enumerate(range(batch_id * batch_size,
                                        min((batch_id + 1) * batch_size, len(self.tiler)))):
            self.add(tile_i, data[data_i])

    def merge(self, unpad: bool = True, argmax: bool = False) -> np.ndarray:
        """Returns final merged data array obtained from added tiles.

        Args:
            unpad (bool): If unpad is True, removes padded elements. Default is True.
            argmax (bool): If argmax is True, the first dimension will be argmaxed. Default is False.

        Returns:
            np.ndarray: Final merged data array obtained from added tiles.
        """
        data = self.data

        if unpad:
            sl = [slice(None, self.tiler.data_shape[i]) for i in range(len(self.tiler.data_shape))]

            if self.logits:
                sl = [slice(None, None, None)] + sl

            data = data[tuple(sl)]

        if argmax:
            data = np.argmax(data, 0)

        return data
