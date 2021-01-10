import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from tqdm.auto import tqdm
from typing import Tuple, List, Union, Callable, Generator
# try:
#     import torch
# except ImportError:
#     pass


class Tiler:

    TILING_MODES = ['constant', 'drop', 'irregular', 'overlap_tile', 'reflect', 'edge', 'wrap']

    # @classmethod
    # def overlap_tile(cls):
    #     pass
    #
    # @classmethod
    # def auto_overlap(cls,
    #                  image_shape: Union[Tuple, List],
    #                  tile_shape: Union[Tuple, List],
    #                  window: str,
    #                  mode: Union[str] = 'constant',
    #                  channel_dimension: Union[int, None] = None,
    #                  offset: Union[int, tuple, List, None] = None,
    #                  constant_value: float = 0.0
    #                  ):
    #     """
    #     Alternative way to create a Tiler object.
    #     Automatically calculates optimal overlap and padding depending on the window function.
    #
    #     :param image_shape:
    #     :param tile_shape:
    #     :param window:
    #     :param mode:
    #     :param channel_dimension:
    #     :param offset:
    #     :param constant_value:
    #     :return:
    #     """
    #
    #     pass

    def __init__(self,
                 image_shape: Union[Tuple, List],
                 tile_shape: Union[Tuple, List],
                 mode: Union[str] = 'constant',
                 channel_dimension: Union[int, None] = None,
                 # offset: Union[int, tuple, List, None] = None,
                 constant_value: float = 0.0,
                 overlap: Union[float, Tuple, List] = 0.0
                 ):
        """
        Tiler precomputes everything for tiling with specified parameters, without requiring actual data.

        :param image_shape: tuple or list
            Image shape, i.e. (1980, 1050, 3) or [512, 512, 512].
            If you have a channel dimension, specify it with channel_dimension keyword.

        :param tile_shape: tuple or list
            Shape of a tile, i.e. (256, 256, 3) or [64, 64, 64].
            Tile must have same the number of dimensions as data.
            # TODO should it be any size?

        :param mode: str # TODO allow to pass Callable too?
            Mode defines how the data will be tiled.

            One of the following string values: # TODO or a user supplied function?
                `constant` (default)
                    Pads tile with constant value to match tile_shape.
                    Set the value with keyword 'constant_value'.
                'drop'
                    Do not return tiles that are smaller than tile_shape.
                'irregular'
                    Tiles can be smaller than tile_shape are.
                `reflect`
                    Pads tile with the reflection of values along each axis.
                `edge`
                    Pads tile with the edge values of data.
                `wrap`
                    Pads the tile with the wrap of the vector along the axis.
                    The first values are used to pad the end and the end values
                    are used to pad the beginning.

                # TODO other padding modes
                # `maximum`
                #     Pads tile with the maximum value of each axis.
                # `minimum`
                #     Pads tile with the minimum value of each axis.
                # `mean`
                #     Pads tile with the mean value of each axis.
                # `median`
                #     Pads tile with the median value of each axis.
                # TODO callable padding?
                # <function>
                #     The function accepts the tile and returns the padded tile.

        :param channel_dimension: int or None
            Used to specify the channel dimension.
            Channel dimension is treated differently from other dimensions.
            The channel dimension will never be tiled.
            Often it is the last or the first dimension.
            Default is None, no channel dimension in the data.

        # :param offset: int or tuple or list or None
        #     # TODO: does not work yet!
        #     Used to add (padable if negative) offset to the dimensions.
        #     If int, the same offset will be applied on each dimension.
        #     If tuple or float, must have the same number of dimensions as tile_shape.
        #     If None, adds a negative offset equal to half of tile_shape.
        #     Default is None.

        :param constant_value: float
            Used in 'constant' mode. The value to set the padded values for each axis.

        :param overlap: int, float or tuple or list
            If int, the same overlap in each dimension.
            If float, percentage of a tile_size to use for overlap (from 0.0 to 1.0).
            If tuple or list, size of the overlap in. Must be smaller than tile_shape.
            Default is 0.0.
        """

        # Image and tile shapes
        self.image_shape = np.asarray(image_shape).astype(int)
        self.tile_shape = np.asarray(tile_shape).astype(int)
        if (self.tile_shape <= 0).any() or (self.image_shape <= 0).any():
            raise ValueError('Shapes must be tuple or lists of positive numbers.')
        if self.tile_shape.size != self.image_shape.size:
            raise ValueError('Tile and data shapes must have the same length. '
                             'If your array has a channel dimension, specify it in `channel_dimension`.')

        # Tiling mode
        self.mode = mode
        if self.mode not in self.TILING_MODES:
            raise ValueError('Unsupported tiling mode, please check docs.')

        # Channel dimension
        self.channel_dimension = channel_dimension
        if self.channel_dimension and ((self.channel_dimension < 0)
                                       or (self.channel_dimension > len(self.image_shape))):
            raise ValueError(f'Specified channel dimension is out of bounds '
                             f'(should be from 0 to {len(self.image_shape)}).')

        # # Offset
        # if offset is None:
        #     # Default offset is negative half the tile_shape
        #     self.offset = (self.tile_shape // 2)
        # elif isinstance(offset, int):
        #     # Int offset applies the same offset to each dimension
        #     self.offset = np.array([offset for _ in self.tile_shape])
        # else:
        #     if self.offset.size != self.tile_shape.size:
        #         raise ValueError('Offset and tile shapes must have the same length. '
        #                          'If your array has a channel dimension, specify it in `channel_dimension`.')
        # if self.channel_dimension:
        #     self.offset[self.channel_dimension] = 0

        # Constant value used for `constant` tiling mode
        self.constant_value = constant_value

        # Overlap
        self.overlap = overlap
        if isinstance(self.overlap, float) and (self.overlap < 0 or self.overlap > 1.0):
            raise ValueError('Overlap, if float, must be in range of 0.0 (0%) to 1.0 (100%).')
        if (isinstance(self.overlap, list) or isinstance(self.overlap, tuple)) \
                and (np.any((self.tile_shape - np.array(self.overlap)) <= 0)):
            raise ValueError('Overlap size much be smaller than tile_shape.')

        # Tiling points and sizes calculations
        self._n_dim = len(image_shape)
        if isinstance(self.overlap, list) or isinstance(self.overlap, tuple):
            # overlap is given
            self._tile_overlap: np.ndarray = np.array(self.overlap).astype(int)
        elif isinstance(self.overlap, int):
            # int overlap applies the same overlap to each dimension
            self._tile_overlap: np.ndarray = np.array([self.overlap for _ in self.tile_shape])
        elif isinstance(self.overlap, float):
            # compute overlap
            self._tile_overlap: np.ndarray = np.ceil(self.overlap * self.tile_shape).astype(int)
        else:
            raise ValueError('Unsupported overlap mode (not float, int, list or tuple).')

        self._tile_step: np.ndarray = (self.tile_shape - self._tile_overlap).astype(int)  # tile step
        self._tile_slices = [slice(None, None, step) for i, step in enumerate(self._tile_step) if step != 0]

        # if channel dimension is given, set tile_step of that dimension to 0
        if self.channel_dimension is not None:
            self._tile_step[self.channel_dimension] = 0
        self._tile_points = [
            list(range(0, image_shape[d] - self._tile_overlap[d], self._tile_step[d]))
            if self._tile_step[d] != 0 else [0]
            for d in range(self._n_dim)
        ]
        self._new_shape = [x[-1] + self.tile_shape[i] for i, x in enumerate(self._tile_points)]
        self._shape_diff = self._new_shape - self.image_shape

        # Drop mode: delete points that would create patches that are smaller than tile_size
        if self.mode == 'drop':
            # delete points that would create patches smaller than tile_size
            for d, x in enumerate(self._shape_diff):
                if 0 < x < self.tile_shape[d]:
                    del self._tile_points[d][-1]

            # recalculate new shape and shape diff
            self._new_shape = [x[-1] + self.tile_shape[i] for i, x in enumerate(self._tile_points)]
            self._shape_diff = self._new_shape - self.image_shape

        # Tile indexing
        # Underneath, the actual tiling is done with numpy's as_strided (returns view = O(1))
        # Returned strided array will be 2n-dimensional with first n being indexing dimensions
        # and last n dimensions contain actual data. Reshaping view to a list of patches
        # would mean copying data (and losing all benefits of view). To avoid that, we have a proxy array
        # that is basically a mapping from 1D (0 to N tiles) to ND tiles.

        # context to remove division by zero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            self._indexing_shape = ((self._new_shape - self.tile_shape) // self._tile_step) + 1

        self._tile_index = np.array(np.meshgrid( *[ np.arange(0, x) for x in self._indexing_shape] ))
        self._tile_index = self._tile_index.T.reshape(-1, self._n_dim)  # reshape into (tile_id, coordinates)
        self.n_tiles = len(self._tile_index)

        # Tile sampling
        self._tile_sample_shapes = np.tile(self.tile_shape, (*self._indexing_shape, 1))
        # Most of the tiles should be full self.tile_shape, but the ones on the edges will probably be out-of-bounds.
        # The problem with view is that there is no OOB checks. We have to keep in mind how many voxels to sample.
        # Border tiles can have shape <tile_shape, so for each last row, column etc. we need to subtract difference.
        for dim, diff in zip(range(self._n_dim), self._shape_diff):
            i = [slice(None)] * self._n_dim
            i[dim] = self._indexing_shape[dim] - 1

            j = [slice(None)] * (self._n_dim - 1)
            j.append(dim)

            # shapes for that dimension elements
            self._tile_sample_shapes[tuple(i)][tuple(j)] -= diff

        # Reshape sample shapes to the same indexing as tiles
        self._tile_sample_shapes = self._tile_sample_shapes.reshape(-1, self._n_dim)

    def __len__(self) -> int:
        """ Returns number of tiles produced by tiling. """
        return self.n_tiles

    def __repr__(self) -> str:
        return f'{list(self.tile_shape)} tiler for data of shape {list(self.image_shape)}:' \
               f'\n\tNew shape: {self._new_shape}' \
               f'\n\tOverlap: {self.overlap}' \
               f'\n\tStep: {list(self._tile_step)}' \
               f'\n\tMode: {self.mode}' \
               f'\n\tChannel dimension: {self.channel_dimension}'

    def __call__(self, data: np.ndarray, progress_bar: bool = False,
                 batch_size: int = 1, drop_last: bool = False) -> \
            Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterate through tiles of the given data array.

        :param data: np.ndarray
            The data array on which the tiling will be performed.

        :param progress_bar: bool
            Whether to show the progress bar or not. Uses tqdm package.

        :param batch_size: int
            # TODO
            # If > 1, returns more than one tile
            # If == 1, does not add batch dimension
            Default: 1

        :param drop_last: bool
            # TODO
            # if n_tiles % batch_size != 0 and drop_last == True, drop the last (incomplete) batch
            # else, returns incomplete batch

        :return: yields (int, np.ndarray)
            Returns tuple with int that is the tile_id and np.ndarray tile data.
        """

        # if batch_size <= 0:
        #     raise ValueError(f'Requested batch_size ({batch_size}) is <= 0')
        #
        # for batch_i in tqdm(range(0, self.n_tiles, batch_size),
        #                     desc='Processing', disable=not progress_bar, unit='tile'):
        #
        #     actual_batch_size = batch_i
        #     collated_tiles =
        tiles = self.view_in_tiles(data)
        for tile_i in tqdm(range(self.n_tiles), desc='Tiling', disable=not progress_bar, unit='tile'):
            yield tile_i, self.get_tile(None, tile_i, tiles)

    def view_in_tiles(self, data: np.ndarray) -> np.ndarray:
        """
        Fast (O(1)) tiling of the data with numpy views.
        Slices data into mosaic of tiles.

        :param data: np.ndarray
            Array to be sliced into tiles.

        :return: np.ndarray
            2 * data.ndim -dimensional array.
            First n dimensions are mosaic coordinates, rest n dimensions are the actual data.
        """
        if np.any(np.array(data.shape) != self.image_shape):
            raise ValueError(f'Data must have the same shape as image_shape '
                             f'({data.shape} != {self.image_shape}).')

        # if isinstance(data, np.ndarray):
        tile_strides = data.strides
        indexing_strides = data[tuple(self._tile_slices)].strides
        # elif isinstance(data, torch.Tensor):
        #     tile_strides = np.multiply(data.stride(), data.element_size())
        #     indexing_strides = np.multiply(data[tuple(self._tile_slices)].stride(), data.element_size())
        # else:
        #     raise ValueError(f'Not np.ndarray, but {type(data)}')

        shape = tuple(list(self._indexing_shape) + list(self.tile_shape))
        strides = tuple(list(indexing_strides) + list(tile_strides))

        # if isinstance(data, np.ndarray):
        tiles = ast(data, shape=shape, strides=strides, writeable=False)
        # else:
        #     tiles = torch.as_strided(data, size=shape, stride=strides)
        return tiles

    def get_tile(self, data: Union[np.ndarray, None], tile_id: int, tiles: np.ndarray = None) -> np.ndarray:
        """
        Returns tile content.

        :param data: np.ndarray
            Data from which tile_id-th tile will be taken.

        :param tile_id: int
            Specify which tile to return. Must be smaller than number of tiles.

        :param tiles: np.ndarray
            # TODO for inner use

        :return: np.ndarray
            Content of tile number tile_id. Padded if necessary.
        """

        if (tile_id < 0) or (tile_id >= self.n_tiles):
            raise IndexError(f'Out of bounds, there is no tile {tile_id}.'
                             f'There are {len(self) - 1} tiles, starting from index 0.')

        # get tiles view
        if tiles is None:
            tiles = self.view_in_tiles(data)

        # get the shape that should be sampled from the tile
        sample_shape = self.get_tile_sample_shape(tile_id, with_channel_dim=(self.channel_dimension is not None))

        # get the actual data for the tile
        tile_view = tiles[tuple(self._tile_index[tile_id])]
        # if isinstance(tile_view, np.ndarray):
        tile_data = tile_view[tuple(slice(None, stop) for stop in sample_shape)].copy()
        # elif isinstance(tile_view, torch.Tensor):
        #     tile_data = tile_view[tuple(slice(None, stop) for stop in sample_shape)].clone()
        # else:
        #     raise ValueError(f'Not np.ndarray, but {type(tile_view)}')

        # # if sample_shape is not the same as tile_shape, we need to pad the tile in the given mode
        # if self.channel_dimension is not None:
        #     sample_shape = np.insert(sample_shape, self.channel_dimension, self.tile_shape[self.channel_dimension])
        shape_diff = self.tile_shape - np.array(sample_shape, ndmin=self.tile_shape.ndim)
        if (self.mode != 'irregular') and np.any(shape_diff > 0):
            if self.mode == 'constant':
                tile_data = np.pad(tile_data, list((0, diff) for diff in shape_diff), mode=self.mode,
                                   constant_values=self.constant_value)
            elif self.mode == 'reflect' or self.mode == 'edge' or self.mode == 'wrap':
                tile_data = np.pad(tile_data, list((0, diff) for diff in shape_diff), mode=self.mode)

        return tile_data

    def get_tile_sample_shape(self, tile_id: int, with_channel_dim: bool = False) -> np.ndarray:
        """
        Returns shape of sample for the tile with number tile_id.
        In other words, shape of a sub-hyperrectangle of tile that was sampled from original data.
        For example if (64, 64) tile was actually padded to that size from (40, 40),
        this method will return (40, 40).

        :param tile_id: int
            Tile ID for which to return sample shape.

        :param with_channel_dim: bool
            Whether to return shape with channel dimension or without.

        :return: np.ndarray
            Shape of sample of the tile.
        """

        if (tile_id < 0) or (tile_id >= self.n_tiles):
            raise IndexError(f'Out of bounds, there is no tile {tile_id}. '
                             f'There are {len(self)} tiles, starting from index 0.')

        if self.channel_dimension is not None and not with_channel_dim:
            return self._tile_sample_shapes[tile_id][~(np.arange(self._n_dim) == self.channel_dimension)]
        return self._tile_sample_shapes[tile_id]

    def get_tile_bbox_position(self, tile_id: int, with_channel_dim: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns diagonal corner coordinates of bounding hyperrectangle of the tile on padded data.
        The first element is the smallest corner, the second is the largest.

        :param tile_id: int
            Specify which tile bounding box must be returned.
            Must be smaller than number of tiles.

        :param with_channel_dim: bool
            Whether to return shape with channel dimension or without.

        :return: (np.ndarray, np.ndarray)
            Smallest and largest corners of the bounding box.
        """

        if (tile_id < 0) or (tile_id >= self.n_tiles):
            raise IndexError(f'Out of bounds, there is no tile {tile_id}. '
                             f'There are {len(self) - 1} tiles, starting from index 0.')

        starting_corner = self._tile_step * self.get_tile_mosaic_position(tile_id, True)
        finish_corner = starting_corner + self.tile_shape
        if self.channel_dimension is not None and not with_channel_dim:
            dim_indices = list(range(self.channel_dimension)) + list(range(self.channel_dimension + 1, len(self._tile_step)))
            starting_corner = starting_corner[dim_indices]
            finish_corner = finish_corner[dim_indices]
        return starting_corner, finish_corner

    def get_tile_mosaic_position(self, tile_id: int, with_channel_dim: bool = False) -> np.ndarray:
        """
        Returns tile position in mosaic (relative to other tiles).

        :param tile_id: int
            Tile ID for which to return mosaic position.

        :param with_channel_dim: bool
            Whether to return position with channel dimension or without.

        :return: np.ndarray
            Tile mosaic position (tile position relative to other tiles).
        """
        if (tile_id < 0) or (tile_id >= self.n_tiles):
            raise IndexError(f'Out of bounds, there is no tile {tile_id}. '
                             f'There are {len(self) - 1} tiles, starting from index 0.')

        if self.channel_dimension is not None and not with_channel_dim:
            return self._tile_index[tile_id][~(np.arange(self._n_dim) == self.channel_dimension)]
        return self._tile_index[tile_id]

    def get_mosaic_shape(self, with_channel_dim: bool = False) -> np.ndarray:
        """
        Return mosaic shape (array/atlas of tiles)

        # :param with_channel_dim: bool
        #     Whether to return shape with channel dimension or without.

        :return: np.ndarray
            Shape of tiles array/atlas.
        """
        if self.channel_dimension is not None and not with_channel_dim:
            return self._indexing_shape[~(np.arange(self._n_dim) == self.channel_dimension)]
        return self._indexing_shape




    #
    # # Merge (processed) tile into accumulator array
    # # Efficient way to accumulate processed images
    # # Supports various windows
    # def merge(self, accumulator: np.ndarray, tile: np.ndarray, tile_id: int, window: str = 'norm'):
    #     if accumulator.shape != self.image_shape:
    #         raise ValueError(f'Accumulator must have the same shape as image_shape '
    #                          f'({accumulator.shape} != {self.image_shape})')
    #     if window not in self.__WINDOWS:
    #         raise ValueError('Unsupported window function, please check docs')
    #
    #
    # # Return border type of the tile
    # def get_tile_border_type(self, tile_id: int):
    #     tile_pos = self._tile_index[tile_id]
    #     tile_n_around = self._tile_border_types[tuple(tile_pos)]
    #     min_max = tile_pos ==  min(tile_pos)
    #     return
    #
    #
    #
    # # Return
    # def is_corner_tile(self, tile_id: int) -> bool:
    #     pass
    #
    # def is_edge_tile(self, tile_id: int) -> bool:
    #     pass
    #
    # # corners
    # # number of corners: 2^n_dim, permutations of all corners
    # # corners direction?
    #
    # # edges
    # #
    # # import numpy as np
    # #
    # # def edge_mask(x):
    # #     mask = np.ones(x.shape, dtype=bool)
    # #     mask[x.ndim * (slice(1, -1),)] = False
    # #     return mask
    # #
    # # x = np.random.rand(4, 5)
    # # edge_mask(x)
    # # # array([[ True,  True,  True,  True,  True],
    # # #        [ True, False, False, False,  True],
    # # #        [ True, False, False, False,  True],
    # # #        [ True,  True,  True,  True,  True]], dtype=bool)
    #
    #
    # #
    # #
    # # def _precompute_window_type(self):
    # #
    # #
    # #
    # # def _get_corners(self) -> np.ndarray:
    # #     corners = a[tuple(slice(None, None, j - 1) for j in a.shape)]
    # #
    # #
    # # def how_many_edges_touching(self, tile_id: int) -> int:
    # #
    # #
    # # def is_border_tile(self, tile_id: int):
    # #     # edge tile will have at least one min or max value in any dimension
    # #     if self.how_many_edges_touching(tile_id) > 0:
    # #         return True
    # #     else:
    # #         return False
    # #
    # # def is_corner_tile(self, tile_id: int):
    # #     # corner tile will have ndim min or max values in any dimensions
    # #     if self.how_many_edges_touching(tile_id) == self._n_dim:
    # #         return True
    # #     else:
    # #         return False
    # #
    # #
    # # def a(self):
    # #     # Define types of possible tiles
    # #
    # #     # 2D case
    # #     # *---------*
    # #     # |1   5   2|
    # #     # |7   9   8|
    # #     # |3   6   4|
    # #     # *---------*
    # #
    # #
    # #     #    +--------+
    # #     #   /        /|
    # #     #  /        / |
    # #     # +--------+  |
    # #     # |        |  |
    # #     # |        |  +
    # #     # |        | /
    # #     # |        |/
    # #     # +--------+
    # #
    # #     # Calculate which tiles are border tiles
    # #
    # #     # self._border_tiles = [tile for tile in self._tiles if tile]
    # #     self._border_tiles = []
    # #     for i in range(len(self._tiles)):
    # #         for dim, x in enumerate(self._tiles[i]):
    # #             if x == self._tile
    # #
    # #         if np.any([True for x in self._tiles[i] if ])
    # #
    # #     for tile in self._tiles:
    # #         # check each dimension and if it is _tile_ends
    # #         if np.any()
    # #         [x for x in tile]
    # #         for dim in tile:
    # #             if tile[dim]
    #
    # # def merge(self, images, window: str = None, crop_padding: bool = True):
    # #     pass
