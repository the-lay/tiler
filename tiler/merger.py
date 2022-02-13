from typing import Union, Tuple, List, Optional
import warnings

import numpy as np
import numpy.typing as npt

from tiler import Tiler
from tiler._windows import get_window


class Merger:

    SUPPORTED_WINDOWS = [
        "boxcar",
        "triang",
        "blackman",
        "hamming",
        "hann",
        "bartlett",
        "parzen",
        "bohman",
        "blackmanharris",
        "nuttall",
        "barthann",
        "overlap-tile",
    ]
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
    - 'parzen'  
        Parzen window.
    - 'bohman'  
        Bohman window.
    - 'blackmanharris'  
        Minimum 4-term Blackman-Harris window.
    - 'nuttall'  
        Minimum 4-term Blackman-Harris window according to Nuttall.
    - 'barthann'  
        Bartlett-Hann window.    
    - 'overlap-tile'  
        Creates a boxcar window for the non-overlapping, middle part of tile, and zeros everywhere else.
        Requires applying padding calculated with `Tiler.calculate_padding()` for correct results.
        (based on Ronneberger et al. 2015, U-Net paper)
    """

    def __init__(
        self,
        tiler: Tiler,
        window: Union[None, str, np.ndarray] = None,
        logits: int = 0,
        save_visits: bool = True,
        data_dtype: npt.DTypeLike = np.float32,
        weights_dtype: npt.DTypeLike = np.float32,
    ):
        """Merger holds cumulative result buffers for merging tiles created by a given Tiler
        and the window function that is applied to added tiles.

        There are two required np.float64 buffers: `self.data` and `self.weights_sum`
        and one optional np.uint32 `self.data_visits` (see below `save_visits` argument).

        TODO:
            - generate window depending on tile border type
                # some reference for the future borders generation
                # 1d = 3 types of tiles: 2 corners and middle
                # 2d = 9 types of tiles: 4 corners, 4 tiles with 1 edge and middle
                # 3d = 25 types of tiles: 8 corners, 12 tiles with 2 edges, 6 tiles with one edge and middle
                # corners: 2^ndim
                # tiles: 2*ndim*nedges

        Args:
            tiler (Tiler): Tiler with which the tiles were originally created.

            window (None, str or np.ndarray): Specifies which window function to use for tile merging.
                Must be one of `Merger.SUPPORTED_WINDOWS` or a numpy array with the same size as the tile.
                Default is None which creates a boxcar window (constant 1s).

            logits (int): Specify whether to add logits dimensions in front of the data array. Default is `0`.

            save_visits (bool): Specify whether to save which elements has been modified and how many times in
                `self.data_visits`. Can be disabled to save some memory. Default is `True`.

            data_dtype (np.dtype): Specify data type for data buffer that stores cumulative result.
                Default is `np.float32`.

            weights_dtype (np.dtype): Specify data type for weights buffer that stores cumulative weights and window array.
                If you don't need precision but would rather save memory you can use `np.float16`.
                Likewise, on the opposite, you can use `np.float64`.
                Default is `np.float32`.

        """

        self.tiler = tiler

        # Logits support
        if not isinstance(logits, int) or logits < 0:
            raise ValueError(
                f"Logits must be an integer 0 or a positive number ({logits})."
            )
        self.logits = int(logits)

        # Generate data and normalization arrays
        self.data = self.data_visits = self.weights_sum = None
        self.data_dtype = data_dtype
        self.weights_dtype = weights_dtype
        self.reset(save_visits)

        # Generate window function
        self.window = None
        self.set_window(window)

    def _generate_window(self, window: str, shape: Union[Tuple, List]) -> np.ndarray:
        """Generate n-dimensional window according to the given shape.
        Adapted from: https://stackoverflow.com/a/53588640/1668421

        Args:
            window (str): Specifies window function. Must be one of `Merger.SUPPORTED_WINDOWS`.

            shape (tuple or list): Shape of the requested window.

        Returns:
            np.ndarray: n-dimensional window of the given shape and function
        """

        w = np.ones(shape, dtype=self.weights_dtype)
        overlap = self.tiler._tile_overlap
        for axis, length in enumerate(shape):
            if axis == self.tiler.channel_dimension:
                # channel dimension should have weight of 1 everywhere
                win = get_window("boxcar", length)
            else:
                if window == "overlap-tile":
                    axis_overlap = overlap[axis] // 2
                    win = np.zeros(length)
                    win[axis_overlap:-axis_overlap] = 1
                else:
                    win = get_window(window, length)

            for i in range(len(shape)):
                if i == axis:
                    continue
                else:
                    win = np.stack([win] * shape[i], axis=i)

            w *= win.astype(self.weights_dtype)

        return w

    def set_window(self, window: Union[None, str, np.ndarray] = None) -> None:
        """Sets window function depending on the given window function.

        Args:
            window (None, str or np.ndarray): Specifies which window function to use for tile merging.
                Must be one of `Merger.SUPPORTED_WINDOWS` or a numpy array with the same size as the tile.
                If passed None sets a boxcar window (constant 1s).

        Returns:
            None
        """

        # Warn user that changing window type after some elements were already visited is a bad idea.
        if np.count_nonzero(self.data_visits):
            warnings.warn(
                "You are setting window type after some elements were already added."
            )

        # Default window is boxcar
        if window is None:
            window = "boxcar"

        # Generate or set a window function
        if isinstance(window, str):
            if window not in self.SUPPORTED_WINDOWS:
                raise ValueError("Unsupported window, please check docs")
            self.window = self._generate_window(window, self.tiler.tile_shape)
        elif isinstance(window, np.ndarray):
            if not np.array_equal(window.shape, self.tiler.tile_shape):
                raise ValueError(
                    f"Window function must have the same shape as tile shape."
                )
            self.window = window.astype(self.weights_dtype)
        else:
            raise ValueError(
                f"Unsupported type for window function ({type(window)}), expected str or np.ndarray."
            )

    def reset(self, save_visits: bool = True) -> None:
        """Reset data, weights and optional data_visits buffers.

        Should be done after finishing merging full tile set and before starting processing the next tile set.

        Args:
            save_visits (bool): Specify whether to save which elements has been modified and how many times in
                `self.data_visits`. Can be disabled to save some memory. Default is `True`.

        Returns:
            None
        """

        padded_data_shape = self.tiler._new_shape

        # Image holds sum of all processed tiles multiplied by the window
        if self.logits:
            self.data = np.zeros(
                (self.logits, *padded_data_shape), dtype=self.data_dtype
            )
        else:
            self.data = np.zeros(padded_data_shape, dtype=self.data_dtype)

        # Data visits holds the number of times each element was assigned
        if save_visits:
            self.data_visits = np.zeros(
                padded_data_shape, dtype=np.uint32
            )  # uint32 ought to be enough for anyone :)

        # Total data window (weight) coefficients
        self.weights_sum = np.zeros(padded_data_shape, dtype=self.weights_dtype)

    def add(self, tile_id: int, data: np.ndarray) -> None:
        """Adds `tile_id`-th tile into Merger.

        Args:
            tile_id (int): Specifies which tile it is.

            data (np.ndarray): Specifies tile data.

        Returns:
            None
        """
        if tile_id < 0 or tile_id >= len(self.tiler):
            raise IndexError(
                f"Out of bounds, there is no tile {tile_id}. "
                f"There are {len(self.tiler)} tiles, starting from index 0."
            )

        data_shape = np.array(data.shape)
        expected_tile_shape = (
            ((self.logits,) + tuple(self.tiler.tile_shape))
            if self.logits > 0
            else tuple(self.tiler.tile_shape)
        )

        if self.tiler.mode != "irregular":
            if not np.all(np.equal(data_shape, expected_tile_shape)):
                raise ValueError(
                    f"Passed data shape ({data_shape}) "
                    f"does not fit expected tile shape ({expected_tile_shape})."
                )
        else:
            if not np.all(np.less_equal(data_shape, expected_tile_shape)):
                raise ValueError(
                    f"Passed data shape ({data_shape}) "
                    f"must be less or equal than tile shape ({expected_tile_shape})."
                )

        # Select coordinates for data
        shape_diff = expected_tile_shape - data_shape
        a, b = self.tiler.get_tile_bbox(tile_id, with_channel_dim=True)

        sl = [slice(x, y - shape_diff[i]) for i, (x, y) in enumerate(zip(a, b))]
        win_sl = [
            slice(None, -diff) if (diff > 0) else slice(None, None)
            for diff in shape_diff
        ]

        if self.logits > 0:
            self.data[tuple([slice(None, None, None)] + sl)] += (
                data * self.window[tuple(win_sl[1:])]
            )
            self.weights_sum[tuple(sl)] += self.window[tuple(win_sl[1:])]
        else:
            self.data[tuple(sl)] += data * self.window[tuple(win_sl)]
            self.weights_sum[tuple(sl)] += self.window[tuple(win_sl)]

        if self.data_visits is not None:
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
            raise IndexError(
                f"Out of bounds. There are {n_batches} batches of {batch_size}, starting from index 0."
            )

        # add each tile in a batch with computed tile_id
        for data_i, tile_i in enumerate(
            range(
                batch_id * batch_size, min((batch_id + 1) * batch_size, len(self.tiler))
            )
        ):
            self.add(tile_i, data[data_i])

    def _unpad(
        self, data: np.ndarray, extra_padding: Optional[List[Tuple[int, int]]] = None
    ):
        """Slices/unpads data according to merger and tiler settings, as well as additional padding.

        Args:
            data (np.ndarray): Data to be sliced.

            extra_padding (tuple of tuples of two ints, optional): Specifies padding that was applied to the data.
                Number of values padded to the edges of each axis.
                ((before_1, after_1), … (before_N, after_N)) unique pad widths for each axis.
                Default is None.
        """
        if extra_padding:
            sl = [
                slice(pad_from, shape - pad_to)
                for shape, (pad_from, pad_to) in zip(
                    self.tiler.data_shape, extra_padding
                )
            ]
        else:
            sl = [
                slice(None, self.tiler.data_shape[i])
                for i in range(len(self.tiler.data_shape))
            ]

        # if merger has logits dimension, add another slicing in front
        if self.logits:
            sl = [slice(None, None, None)] + sl

        return data[tuple(sl)]

    def merge(
        self,
        unpad: bool = True,
        extra_padding: Optional[List[Tuple[int, int]]] = None,
        argmax: bool = False,
        normalize_by_weights: bool = True,
        dtype: Optional[npt.DTypeLike] = None,
    ) -> np.ndarray:
        """Returns merged data array obtained from added tiles.

        Args:
            unpad (bool): If unpad is True, removes padded array elements. Default is True.

            extra_padding (tuple of tuples of two ints, optional): Specifies padding that was applied to the data.
                Number of values padded to the edges of each axis.
                ((before_1, after_1), … (before_N, after_N)) unique pad widths for each axis.
                Default is None.

            argmax (bool): If argmax is True, the first dimension will be argmaxed.
                Useful when merger is initialized with `logits=True`.
                Default is False.

            normalize_by_weights (bool): If normalize is True, the accumulated data will be divided by weights.
                Default is True.

            dtype (np.dtype, optional): Specify dtype for the final merged output.
                If None, uses `data_dtype` specified when Merger was initialized.
                Default is None.

        Returns:
            np.ndarray: Final merged data array obtained from added tiles.
        """

        data = self.data

        if normalize_by_weights:
            # ignoring division by zero
            # alternatively, set values < atol to 1
            # https://github.com/the-lay/tiler/blob/46e948bb2bd7a909e954baf87a0c15b384109fde/tiler/merger.py#L314
            # TODO check which way is better
            #  ignoring should be more precise without atol
            #  but can hide other errors
            with np.errstate(divide="ignore", invalid="ignore"):
                data = np.nan_to_num(data / self.weights_sum)

        if unpad:
            data = self._unpad(data, extra_padding)

        if argmax:
            data = np.argmax(data, 0)

        if dtype is not None:
            return data.astype(dtype)
        else:
            return data.astype(self.data_dtype)
