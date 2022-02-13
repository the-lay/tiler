import warnings
import itertools

import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Tuple, List, Union, Generator, Callable


class Tiler:
    TILING_MODES = ["constant", "drop", "irregular", "reflect", "edge", "wrap"]
    r"""
    Supported tiling modes:
    - `constant` (default)  
        If a tile is smaller than `tile_shape`, pad it with the constant value along each axis to match `tile_shape`.
        Set the value with the keyword `constant_value`.  
    - `drop`  
        If a tile is smaller than `tile_shape` in any of the dimensions, ignore it. Can result in zero tiles.
    - `irregular`  
        Allow tiles to be smaller than `tile_shape`.
    - `reflect`  
        If a tile is smaller than `tile_shape`,
        pad it with the reflection of values along each axis to match `tile_shape`.
    - `edge`  
        If a tile is smaller than `tile_shape`,
        pad it with the edge values of data along each axis to match `tile_shape`.
    - `wrap`  
        If a tile is smaller than `tile_shape`,
        pad it with the wrap of the vector along each axis to match `tile_shape`.
        The first values are used to pad the end and the end values are used to pad the beginning.
    """

    def __init__(
        self,
        data_shape: Union[Tuple, List, np.ndarray],
        tile_shape: Union[Tuple, List, np.ndarray],
        overlap: Union[int, float, Tuple, List, np.ndarray] = 0,
        channel_dimension: Optional[int] = None,
        mode: str = "constant",
        constant_value: float = 0.0,
    ):
        """Tiler class precomputes everything for tiling with specified parameters, without actually slicing data.
        You can access tiles individually with `Tiler.get_tile()` or with an iterator, both individually and in batches,
        with `Tiler.iterate()` (or the alias `Tiler.__call__()`).

        Args:
            data_shape (tuple, list or np.ndarray): Input data shape, e.g. (1920, 1080, 3), [512, 512, 512] or np.ndarray([3, 1024, 768]).
                If there is a channel dimension, it should be included in the shape.

            tile_shape (tuple, list or np.ndarray): Shape of a tile, e.g. (256, 256, 3), [64, 64, 64] or np.ndarray([3, 128, 128]).
                Tile must have the same number of dimensions as data or less.
                If less, the shape will be automatically prepended with ones to match data_shape size.

            overlap (int, float, tuple, list or np.ndarray): Specifies overlap between tiles.
                If integer, the same overlap of overlap pixels applied in each dimension, except channel_dimension.
                If float, percentage of a tile_shape to overlap (from 0.0 to 1.0), except channel_dimension.
                If tuple, list or np.ndarray, explicit size of the overlap (must be smaller than tile_shape in each dimension).
                Default is `0`.

            channel_dimension (int, optional): Specifies which axis is the channel dimension that will not be tiled.
                Usually it is the last or the first dimension of the array.
                Negative indexing (`-len(data_shape)` to `-1` inclusive) is allowed.
                Default is `None`, no channel dimension in the data.

            mode (str): Defines how the data will be tiled.
                Must be one of the supported `Tiler.TILING_MODES`. Defaults to `"constant"`.

            constant_value (float): Specifies the value of padding when `mode='constant'`.
                Default is `0.0`.
        """

        self.recalculate(
            data_shape=data_shape,
            tile_shape=tile_shape,
            overlap=overlap,
            channel_dimension=channel_dimension,
            mode=mode,
            constant_value=constant_value,
        )

    def recalculate(
        self,
        data_shape: Optional[Union[Tuple, List, np.ndarray]] = None,
        tile_shape: Optional[Union[Tuple, List, np.ndarray]] = None,
        overlap: Optional[Union[int, float, Tuple, List, np.ndarray]] = None,
        channel_dimension: Optional[int] = None,
        mode: Optional[str] = None,
        constant_value: Optional[float] = None,
    ) -> None:
        """Recalculates tiling for new given settings.
        If a passed value is None, use previously given value.

        For more information about each argument see `Tiler.__init__()` documentation.
        """

        # Data and tile shapes
        if data_shape is not None:
            self.data_shape = np.asarray(data_shape, dtype=np.int64)
        if tile_shape is not None:
            self.tile_shape = np.atleast_1d(np.asarray(tile_shape, dtype=np.int64))

            # Append ones to match data_shape size
            if self.tile_shape.size < self.data_shape.size:
                size_difference = self.data_shape.size - self.tile_shape.size
                self.tile_shape = np.insert(
                    arr=self.tile_shape, obj=0, values=np.ones(size_difference), axis=0
                )
                warnings.warn(
                    f"Tiler automatically adjusted tile_shape from {tuple(tile_shape)} to {tuple(self.tile_shape)}."
                )
        self._n_dim: int = len(self.data_shape)
        if (self.tile_shape <= 0).any() or (self.data_shape <= 0).any():
            raise ValueError(
                "Tile and data shapes must be tuple, list or ndarray of positive integers."
            )
        if self.tile_shape.size != self.data_shape.size:
            raise ValueError(
                "Tile shape must have less or equal number of elements compared to the data shape. "
                "If less, your tile shape will be prepended with ones to match the data shape, "
                "e.g. data_shape=(28, 28), tile_shape=(28) -> tile_shape=(1, 28)."
            )

        # Tiling mode
        if mode is not None:
            self.mode = mode
        if self.mode not in self.TILING_MODES:
            raise ValueError(
                f"{self.mode} is an unsupported tiling mode, please check the documentation."
            )

        # Constant value used for constant tiling mode
        if constant_value is not None:
            self.constant_value = constant_value

        # Channel dimension
        # Channel dimension can be None which means we need to check for init too
        if not hasattr(self, "channel_dimension") or channel_dimension is not None:
            self.channel_dimension = channel_dimension
        if self.channel_dimension:
            if (self.channel_dimension >= self._n_dim) or (
                self.channel_dimension < -self._n_dim
            ):
                raise ValueError(
                    f"Specified channel dimension is out of bounds "
                    f"(should be None or an integer from {-self._n_dim} to {self._n_dim - 1})."
                )
            if self.channel_dimension < 0:
                # negative indexing
                self.channel_dimension = self._n_dim + self.channel_dimension

        # Overlap and step
        if overlap is not None:
            self.overlap = overlap
        if isinstance(self.overlap, float):
            if self.overlap < 0 or self.overlap > 1.0:
                raise ValueError(
                    "Float overlap must be in range of 0.0 (0%) to 1.0 (100%)."
                )

            self._tile_overlap: np.ndarray = np.ceil(
                self.overlap * self.tile_shape
            ).astype(int)
            if self.channel_dimension is not None:
                self._tile_overlap[self.channel_dimension] = 0

        elif isinstance(self.overlap, int):
            tile_shape_without_channel = self.tile_shape[
                np.arange(self._n_dim) != self.channel_dimension
            ]
            if self.overlap < 0 or np.any(self.overlap >= tile_shape_without_channel):
                raise ValueError(
                    f"Integer overlap must be in range of 0 to {np.max(tile_shape_without_channel)}"
                )

            self._tile_overlap: np.ndarray = np.array(
                [self.overlap for _ in self.tile_shape]
            )
            if self.channel_dimension is not None:
                self._tile_overlap[self.channel_dimension] = 0

        elif (
            isinstance(self.overlap, list)
            or isinstance(self.overlap, tuple)
            or isinstance(self.overlap, np.ndarray)
        ):
            if np.any(np.array(self.overlap) < 0) or np.any(
                self.overlap >= self.tile_shape
            ):
                raise ValueError("Overlap size must be smaller than tile_shape.")

            self._tile_overlap: np.ndarray = np.array(self.overlap).astype(int)

        else:
            raise ValueError(
                "Unsupported overlap mode (not float, int, list, tuple or np.ndarray)."
            )

        self._tile_step: np.ndarray = (self.tile_shape - self._tile_overlap).astype(
            int
        )  # tile step

        # Calculate mosaic (collection of tiles) shape
        div, mod = np.divmod(
            [self.data_shape[d] - self._tile_overlap[d] for d in range(self._n_dim)],
            self._tile_step,
        )
        if self.mode == "drop":
            self._indexing_shape = div
        else:
            self._indexing_shape = div + (mod != 0)
        if self.channel_dimension is not None:
            self._indexing_shape[self.channel_dimension] = 1

        # Calculate new shape assuming tiles are padded
        if self.mode == "irregular":
            self._new_shape = self.data_shape
        else:
            self._new_shape = (
                self._indexing_shape * self._tile_step
            ) + self._tile_overlap
        self._shape_diff = self._new_shape - self.data_shape
        if self.channel_dimension is not None:
            self._shape_diff[self.channel_dimension] = 0

        # If channel dimension is given, set tile_step of that dimension to 0
        if self.channel_dimension is not None:
            self._tile_step[self.channel_dimension] = 0

        # Tile indexing
        self._tile_index = np.vstack(
            np.meshgrid(*[np.arange(0, x) for x in self._indexing_shape], indexing="ij")
        )
        self._tile_index = self._tile_index.reshape(self._n_dim, -1).T
        self.n_tiles = len(self._tile_index)

        if self.n_tiles == 0:
            warnings.warn(
                f"Tiler (mode={mode}, overlap={overlap}) will split data_shape {data_shape} "
                f"into zero tiles (tile_shape={tile_shape})."
            )

    def __len__(self) -> int:
        """
        Returns:
             int: Number of tiles in the mosaic.
        """
        return self.n_tiles

    def __repr__(self) -> str:
        """
        Returns:
            str: String representation of the object.
        """
        return (
            f"Tiler split {list(self.data_shape)} data into {len(self)} tiles of {list(self.tile_shape)}."
            f"\n\tMosaic shape: {list(self._indexing_shape)}"
            f"\n\tPadded shape: {list(self._new_shape)}"
            f"\n\tTile overlap: {self.overlap}"
            f"\n\tElement step: {list(self._tile_step)}"
            f"\n\tMode: {self.mode}"
            f"\n\tChannel dimension: {self.channel_dimension}"
        )

    def __call__(
        self,
        data: Union[np.ndarray, Callable[..., np.ndarray]],
        progress_bar: bool = False,
        batch_size: int = 0,
        drop_last: bool = False,
        copy_data: bool = True,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Alias for `Tiler.iterate()`"""
        return self.iterate(data, progress_bar, batch_size, drop_last, copy_data)

    def iterate(
        self,
        data: Union[np.ndarray, Callable[..., np.ndarray]],
        progress_bar: bool = False,
        batch_size: int = 0,
        drop_last: bool = False,
        copy_data: bool = True,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterates through tiles of the given data array. This method can also be accessed by `Tiler.__call__()`.

        Args:
            data (np.ndarray or callable): The data array on which the tiling will be performed. A callable can be
                supplied to load data into memory instead of slicing from an array. The callable should take integers
                as input, the smallest tile corner coordinates and tile size in each dimension, and output numpy array.

                e.g.
                *python-bioformats*
                ```python
                >>> tileSize = 2000
                >>> tiler = Tiler((sizeX, sizeY, sizeC), (tileSize, tileSize, sizeC))
                >>> def reader_func(*args):
                >>>     X, Y, W, H = args[0], args[1], args[3], args[4]
                >>>     return reader.read(XYWH=[X, Y, W, H])
                >>> for t_id, tile in tiler.iterate(reader_func):
                >>>     pass
                ```
                *open-slide*
                ```python
                >>> reader_func = lambda *args: wsi.read_region([args[0], args[1]], 0, [args[3], args[4]])
                >>> for t_id, tile in tiler.iterate(reader_func):
                >>>     pass
                ```

            progress_bar (bool): Specifies whether to show the progress bar or not.
                Uses `tqdm` package.
                Default is `False`.

            batch_size (int): Specifies returned batch size.
                If `batch_size == 0`, return one tile at a time.
                If `batch_size >= 1`, return in batches (returned shape: `[batch_size, *tile_shape]`).
                Default is 0.

            drop_last (bool): Specifies whether to drop last non-full batch.
                Used only when batch_size > 0.
                Default is False.

            copy_data (bool): Specifies whether to copy the tile before returning it.
                If `copy_data == False`, returns a view.
                Default is True.

        Yields:
            (int, np.ndarray): Tuple with integer tile number and array tile data.
        """

        if batch_size < 0:
            raise ValueError(f"Batch size must >= 0, not {batch_size}")

        # return a tile at a time
        if batch_size == 0:
            for tile_i in tqdm(
                range(self.n_tiles), disable=not progress_bar, unit=" tiles"
            ):
                yield tile_i, self.get_tile(data, tile_i, copy_data=copy_data)

        # return in batches
        if batch_size > 0:
            # check for drop_last
            length = (
                (self.n_tiles - (self.n_tiles % batch_size))
                if drop_last
                else self.n_tiles
            )

            for tile_i in tqdm(
                range(0, length, batch_size), disable=not progress_bar, unit=" batches"
            ):
                tiles = np.stack(
                    [
                        self.get_tile(data, x, copy_data=copy_data)
                        for x in range(tile_i, min(tile_i + batch_size, length))
                    ]
                )
                yield tile_i // batch_size, tiles

    def get_tile(
        self,
        data: Union[np.ndarray, Callable[..., np.ndarray]],
        tile_id: int,
        copy_data: bool = True,
    ) -> np.ndarray:
        """Returns an individual tile.

        Args:
            data (np.ndarray or callable): Data from which `tile_id`-th tile will be taken. A callable can be
                supplied to load data into memory instead of slicing from an array. The callable should take integers
                as input, the smallest tile corner coordinates and tile size in each dimension, and output numpy array.

                e.g.
                *python-bioformats*
                ```python
                >>> tileSize = 2000
                >>> tiler = Tiler((sizeX, sizeY, sizeC), (tileSize, tileSize, sizeC))
                >>> def reader_func(*args):
                >>>     X, Y, W, H = args[0], args[1], args[3], args[4]
                >>>     return reader.read(XYWH=[X, Y, W, H])
                >>> tiler.get_tile(reader_func, 0)
                ```
                *open-slide*
                ```python
                >>> reader_func = lambda *args: wsi.read_region([args[0], args[1]], 0, [args[3], args[4]])
                >>> tiler.get_tile(reader_func, 0)
                ```

            tile_id (int): Specifies which tile to return. Must be smaller than the total number of tiles.

            copy_data (bool): Specifies whether returned tile is a copy.
                If `copy_data == False` returns a view.
                Default is True.

        Returns:
            np.ndarray: Content of tile number `tile_id`, padded if necessary.
        """

        if (tile_id < 0) or (tile_id >= self.n_tiles):
            raise IndexError(
                f"Out of bounds, there is no tile {tile_id}."
                f"There are {len(self) - 1} tiles, starting from index 0."
            )

        if (
            isinstance(data, np.ndarray)
            and np.not_equal(data.shape, self.data_shape).any()
        ):
            raise ValueError(
                f"Shape of provided data array ({data.shape}) does not match "
                f"same as Tiler's data_shape ({tuple(self.data_shape)})."
            )

        # get tile data
        tile_corner = self._tile_index[tile_id] * self._tile_step
        # take the lesser of the tile shape and the distance to the edge
        sampling = [
            slice(
                tile_corner[d],
                np.min([self.data_shape[d], tile_corner[d] + self.tile_shape[d]]),
            )
            for d in range(self._n_dim)
        ]

        if callable(data):
            sampling = [x.stop - x.start for x in sampling]
            tile_data = data(*tile_corner, *sampling)
        else:
            tile_data = data[tuple(sampling)]

        if copy_data:
            tile_data = tile_data.copy()

        shape_diff = self.tile_shape - tile_data.shape
        if (self.mode != "irregular") and np.any(shape_diff > 0):
            if self.mode == "constant":
                tile_data = np.pad(
                    tile_data,
                    list((0, diff) for diff in shape_diff),
                    mode=self.mode,
                    constant_values=self.constant_value,
                )
            elif self.mode == "reflect" or self.mode == "edge" or self.mode == "wrap":
                tile_data = np.pad(
                    tile_data, list((0, diff) for diff in shape_diff), mode=self.mode
                )

        return tile_data

    def get_all_tiles(
        self,
        data: Union[np.ndarray, Callable[..., np.ndarray]],
        axis: int = 0,
        copy_data: bool = True,
    ) -> np.ndarray:
        """Returns all tiles joined along a new axis. Does not work for `Tiler.mode = 'irregular'`.

        The `axis` parameter specifies the index of the new axis in the dimensions of the result.
        For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last dimension.

        For more information about `data` and `copy_data` parameters, see `Tiler.get_tile()`.

        Args:
            data (np.ndarray or callable): Data which will be tiled. A callable can be supplied to load data into memory
                instead of slicing from an array. The callable should take integers as input, the smallest tile corner
                coordinates and tile size in each dimension, and output numpy array.

            axis (int): The axis in the result array along which the tiles are stacked.

            copy_data (bool): Specifies whether returned tile is a copy.
                If `copy_data == False` returns a view.
                Default is True.

        Returns:
            np.ndarray: All tiles stacked along a new axis.
        """

        if self.mode == "irregular":
            raise ValueError("get_all_tiles does not support irregular mode")

        return np.stack(
            [self.get_tile(data, x, copy_data=copy_data) for x in range(self.n_tiles)],
            axis=axis,
        )

    def get_tile_bbox(
        self,
        tile_id: int,
        with_channel_dim: bool = False,
        all_corners: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Returns coordinates of the opposite corners of the bounding box (hyperrectangle?) of the tile on padded data.

        Args:
            tile_id (int): Specifies which tile's bounding coordinates will be returned.
                Must be between 0 and the total number of tiles.

            with_channel_dim (bool): Specifies whether to return shape with channel dimension or without.
                Default is False.

            all_corners (bool): If True, returns all vertices of the bounding box.
                Default is False.

        Returns:
            (np.ndarray, np.ndarray): Smallest (bottom-left) and largest (top-right) corners of the bounding box.

            np.ndarray: All corners of the bounding box, if `all_corners=True`.
        """

        if (tile_id < 0) or (tile_id >= self.n_tiles):
            raise IndexError(
                f"Out of bounds, there is no tile {tile_id}. "
                f"There are {len(self) - 1} tiles, starting from index 0."
            )

        # find min and max vertices
        bottom_left_corner = self._tile_step * self.get_tile_mosaic_position(
            tile_id, True
        )
        top_right_corner = bottom_left_corner + self.tile_shape

        # remove channel dimension if not required
        if self.channel_dimension is not None and not with_channel_dim:
            dim_indices = list(range(self.channel_dimension)) + list(
                range(self.channel_dimension + 1, len(self._tile_step))
            )
            bottom_left_corner = bottom_left_corner[dim_indices]
            top_right_corner = top_right_corner[dim_indices]

        # by default, return only min/max vertices
        if not all_corners:
            return bottom_left_corner, top_right_corner

        # otherwise, return all vertices of the bbox
        # inspired by https://stackoverflow.com/a/57065356/1668421
        # but instead create an indexing array from cartesian product of bits
        # and use it to sample intervals
        else:
            n_dim: int = len(bottom_left_corner)  # already channel_dimension adjusted
            mins = np.minimum(bottom_left_corner, top_right_corner)
            maxs = np.maximum(bottom_left_corner, top_right_corner)
            intervals = np.stack([mins, maxs], -1)
            indexing = np.array(list(itertools.product([0, 1], repeat=n_dim)))
            corners = np.stack([intervals[x][indexing.T[x]] for x in range(n_dim)], -1)
            return corners

    def get_tile_mosaic_position(
        self, tile_id: int, with_channel_dim: bool = False
    ) -> np.ndarray:
        """Returns tile position in the mosaic.

        Args:
          tile_id (int): Specifies which tile's mosaic position will be returned. \
            Must be smaller than the total number of tiles.

          with_channel_dim (bool): Specifies whether to return position with channel dimension or without.
            Default is False.

        Returns:
            np.ndarray: Tile mosaic position (tile position relative to other tiles).
        """
        if (tile_id < 0) or (tile_id >= self.n_tiles):
            raise IndexError(
                f"Out of bounds, there is no tile {tile_id}. "
                f"There are {len(self) - 1} tiles, starting from index 0."
            )

        if self.channel_dimension is not None and not with_channel_dim:
            return self._tile_index[tile_id][
                ~(np.arange(self._n_dim) == self.channel_dimension)
            ]
        return self._tile_index[tile_id]

    def get_mosaic_shape(self, with_channel_dim: bool = False) -> np.ndarray:
        """Returns mosaic shape.

        Args:
            with_channel_dim (bool):
                Specifies whether to return shape with channel dimension or without. Defaults to False.

        Returns:
            np.ndarray: Shape of tiles mosaic.
        """
        if self.channel_dimension is not None and not with_channel_dim:
            return self._indexing_shape[
                ~(np.arange(self._n_dim) == self.channel_dimension)
            ]
        return self._indexing_shape

    def calculate_padding(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Calculate a frame padding for the current Tiler parameters.
        The padding is overlap//2 or tile_step//2, whichever is bigger.
        The method returns a tuple (new_shape, padding) where padding is
        ((before_1, after_1), … (before_N, after_N)), unique pad widths for each axis N.

        In the usual workflow, you'd recalculate tiling settings and then apply padding, prior to tiling.
        Then when merging, pass padding to `Merger.merge(extra_padding=padding, ...)`:
        ```python
        >>> tiler = Tiler(...)
        >>> merger = Merger(tiler, ...)
        >>> new_shape, padding = tiler.calculate_padding()
        >>> tiler.recalculate(data_shape=new_shape)
        >>> padded_data = np.pad(data, pad_width=padding, mode="reflect")
        >>> for tile_id, tile in tiler(padded_data):
        >>>     processed_tile = process(tile)
        >>>     merger.add(tile_id, processed_tile)
        >>> final_image = merger.merge(extra_padding=padding)
        ```
        Return:
            np.ndarray: Resulting shape when padding is applied.

            List[Tuple[int, int]]: Calculated padding.
        """

        # Choosing padding
        pre_pad = np.maximum(self._tile_step // 2, self._tile_overlap // 2)
        post_pad = pre_pad + np.mod(self._tile_step, 2)

        new_shape = pre_pad + self.data_shape + post_pad
        padding = list(zip(pre_pad, post_pad))

        return new_shape, padding
