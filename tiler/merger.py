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
        window: Union[str, np.ndarray] = "boxcar",
        ignore_channels: bool = False,
        logits_n: Optional[int] = None,
        logits_dim: int = 0,
        visits_buffer: bool = True,
        data_dtype: npt.DTypeLike = np.float32,
        weights_dtype: npt.DTypeLike = np.float32,
        visits_dtype: npt.DTypeLike = np.uint32,
    ):
        """Merger holds cumulative result buffers for merging tiles created by a given Tiler
        and the window function that is applied to the added tiles.

        There are two required buffers: `self.data` and `self.weights`
        and one optional `self.visits` (enabled by the keyword `visits_buffer`).

        Args:
            tiler (Tiler): Tiler with which the tiles were originally created.

            window (str or np.ndarray): Specifies which window function to use for tile merging.
                Must be one of `Merger.SUPPORTED_WINDOWS` or a numpy array with the same size as the tile.
                Default is "boxcar" which creates a boxcar window (constant 1s).

            ignore_channels (bool): If True, ignores channel dimension set in Tiler and makes Merger expect tiles
                without channel dimensions. Default is `False`.

            logits_n (int, optional): Specifies the number of classes Merger is expected to hold, i.e. whether
                Moreover, if set, makes Merger ignore Tiler's channel dimension.
                Useful for merging multi-class segmentation predictions. Works in combination with `logits_dim`.
                Default is `None`.

            logits_dim (int): If `logits_n` is set, specifies in which dimension logits should be expected.
                Supports negative indexing. Default is `0`.

            visits_buffer (bool): Specifies whether to enable visits buffer which specifies how many times
                each element has been modified. Can be disabled to save memory. Default is `True`.

            data_dtype (np.dtype): Data type for data buffer that stores cumulative result.
                Default is `np.float32`.

            weights_dtype (np.dtype): Data type for window array and weights buffer that stores cumulative weights.
                Can be used for precision-memory tradeoff, i.e. `np.float16` can save some memory but less precise.
                On the opposite, you can use `np.float64` if you require high fpoint precision.
                Default is `np.float32`.

            visits_dtype (np.dtype): Data type for visits buffer. Used only if `visits_buffer` is True.
                Since visits are discrete, uint data types are recommended.
                Default is `np.uint32`, "ought to be enough for anybody".
        """

        self.tiler = tiler
        self.ignore_channels = ignore_channels

        # Logits support
        if logits_n is not None:
            if not isinstance(logits_n, int) or logits_n <= 0:
                raise ValueError(f"Number of logits must be a positive integer")
            if not isinstance(logits_dim, int):
                raise ValueError(f"Logits dimension must be an integer")

            # support negative indexing for logits dimensions
            n_dim = self.tiler._n_dim
            if logits_dim >= n_dim or logits_dim < -n_dim:
                raise ValueError(f"Logits dimension must be from {-n_dim} to {n_dim-1}")
            if logits_dim < 0:
                logits_dim = n_dim + logits_dim

        self.logits_n = logits_n
        self.logits_dim = logits_dim

        # Data, visits and weights buffers
        self.data = self.weights = self.visits = None
        self.data_dtype = data_dtype
        self.weights_dtype = weights_dtype
        self.visits_dtype = visits_dtype
        self.visits_buffer = visits_buffer
        self._expected_tile_shape = None
        self.reset()

        # Window function
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

    def set_window(self, window: Union[str, np.ndarray] = "boxcar") -> None:
        """Sets window function depending on the given window function.

        Args:
            window (str or np.ndarray): Specifies which window function to use for tile merging.
                Must be one of `Merger.SUPPORTED_WINDOWS` or a numpy array with the same size as the tile.
                Default is "boxcar" (constant 1s).

        Returns:
            None
        """

        # Warn user that changing window after some elements were already visited is a bad idea.
        if np.count_nonzero(self.visits):
            warnings.warn(
                "You are setting window type after some elements were already added."
            )

        # Generate or set a window function
        if isinstance(window, str):
            if window not in self.SUPPORTED_WINDOWS:
                raise ValueError(f"Unsupported window {window}, please check docs.")

            self.window = self._generate_window(
                window, self.tiler.tile_shape_wo_channel
            )
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

    def reset(self) -> None:
        """Resets data, weights and optional data_visits buffers, and recalculates expected tile shape.
        Should be called if you want to reuse the same Merger for another image.

        Returns:
            None
        """

        # Data buffer holds sum of all processed tiles multiplied by the window
        # Weights buffer holds sum of window weights coefficients per element
        # Optional visits buffer holds number of times each element was added to Merger
        # Also, calculate expected tile shape
        data_shape = self.tiler._new_shape
        if self.logits_n and self.ignore_channels:
            ds_wo_channels = data_shape[
                np.arange(self.tiler._n_dim) != self.tiler.channel_dimension
            ]

            ds_wo_channels_w_logits = np.insert(
                ds_wo_channels,
                self.logits_dim,
                self.logits_n,
            )

            self._ds_shape = ds_wo_channels_w_logits

            self.data = np.zeros(self._ds_shape, dtype=self.data_dtype)
            self.weights = np.zeros(self._ds_shape, dtype=self.weights_dtype)

            self._expected_tile_shape = np.insert(
                self.tiler.tile_shape_wo_channel,
                self.logits_dim,
                self.logits_n,
            )

            if self.visits_buffer:
                self.visits = np.zeros(self._ds_shape, dtype=self.weights_dtype)

        elif self.logits_n and not self.ignore_channels:
            ds_w_logits = np.insert(data_shape, self.logits_dim, self.logits_n)
            self._ds_shape = ds_w_logits

            self.data = np.zeros(self._ds_shape, dtype=self.data_dtype)
            self.weights = np.zeros(self._ds_shape, dtype=self.weights_dtype)

            self._expected_tile_shape = np.insert(
                self.tiler.tile_shape,
                self.logits_dim,
                self.logits_n,
            )

            if self.visits_buffer:
                self.visits = np.zeros(self._ds_shape, dtype=self.weights_dtype)

        else:
            self._ds_shape = data_shape

            self.data = np.zeros(self._ds_shape, dtype=self.data_dtype)
            self.weights = np.zeros(self._ds_shape, dtype=self.weights_dtype)

            self._expected_tile_shape = self.tiler.tile_shape

            if self.visits_buffer:
                self.visits = np.zeros(self._ds_shape, dtype=self.weights_dtype)

    def add(self, tile_id: int, data: np.ndarray) -> None:
        """Adds `tile_id`-th tile into Merger buffers.

        Args:
            tile_id (int): Tile id
            data (np.ndarray): Tile data

        Returns:
            None
        """
        if tile_id < 0 or tile_id >= len(self.tiler):
            raise IndexError(
                f"Out of bounds, there is no tile {tile_id}. "
                f"There are {len(self.tiler)} tiles, starting from index 0."
            )

        data_shape = np.array(data.shape)

        if self.tiler.mode != "irregular":
            if not np.all(np.equal(data_shape, self._expected_tile_shape)):
                raise ValueError(
                    f"Passed data shape ({data_shape}) "
                    f"does not fit expected tile shape ({self._expected_tile_shape})."
                )
        else:
            if not np.all(np.less_equal(data_shape, self._expected_tile_shape)):
                raise ValueError(
                    f"Passed data shape ({data_shape}) "
                    f"must be less or equal than tile shape ({self._expected_tile_shape})."
                )

        # Find difference between expected tile shape and provided tile shape
        shape_diff = self._expected_tile_shape - data_shape

        # Get tile bbox data buffer coordinates
        a, b = self.tiler.get_tile_bbox(
            tile_id, with_channel_dim=not self.ignore_channels
        )

        # Generate slicing that puts the provided tile into the data buffer
        sl = [slice(x, y - shape_diff[i]) for i, (x, y) in enumerate(zip(a, b))]
        if self.logits_n:
            # add whole axis for logits dimension
            sl.insert(self.logits_dim, slice(None, None, None))

        # Generate window slicing
        win_sl = [
            slice(None, -diff) if (diff > 0) else slice(None, None)
            for i, diff in enumerate(shape_diff)
            if i != self.tiler.channel_dimension
        ]

        # expand dimensions for correct broadcasting
        if self.logits_n:
            win_sl.insert(self.logits_dim, np.newaxis)
        if not self.ignore_channels:
            win_sl.insert(self.tiler.channel_dimension, np.newaxis)

        # Add to data and weights buffers
        self.data[tuple(sl)] += data * self.window[tuple(win_sl)]
        self.weights[tuple(sl)] += self.window[tuple(win_sl)]

        # Add to visits buffer
        if self.visits_buffer:
            self.visits[tuple(sl)] += 1

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
        self,
        data: np.ndarray,
        extra_padding: Optional[List[Tuple[int, int]]] = None,
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
                for shape, (pad_from, pad_to) in zip(self._ds_shape, extra_padding)
            ]
        else:
            sl = [slice(None, i) for i in self._ds_shape]

        return data[tuple(sl)]

    def merge(
        self,
        unpad: bool = True,
        extra_padding: Optional[List[Tuple[int, int]]] = None,
        argmax: Optional[int] = None,
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

            argmax (int, optional): If set, specifies dimension to be argmaxed.
                Useful in combination with `Merger.logits_n` and `Merger.logits_dim`.
                Default is None

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
                data = np.nan_to_num(data / self.weights)

        if unpad:
            data = self._unpad(data, extra_padding)

        if argmax:
            data = np.argmax(data, argmax)

        if dtype is not None:
            return data.astype(dtype)
        else:
            return data.astype(self.data_dtype)
