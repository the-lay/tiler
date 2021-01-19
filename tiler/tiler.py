import numpy as np
from tqdm.auto import tqdm
from typing import Tuple, List, Union, Callable, Generator
# try:
#     import torch
# except ImportError:
#     pass


class Tiler:

    TILING_MODES = ['constant', 'drop', 'irregular', 'reflect', 'edge', 'wrap']

    def __init__(self,
                 image_shape: Union[Tuple, List],
                 tile_shape: Union[Tuple, List],
                 overlap: Union[int, float, Tuple, List] = 0,
                 channel_dimension: Union[int, None] = None,
                 mode: str = 'constant',
                 constant_value: float = 0.0
                 ):
        """
        Tiler precomputes everything for tiling with specified parameters, without requiring actual data.

        :param image_shape: tuple or list
            Image shape, i.e. (1980, 1050, 3) or [512, 512, 512].
            Note: include channel dimension too, and specify which axis that is with channel_dimension keyword.

        :param tile_shape: tuple or list
            Shape of a tile, i.e. (256, 256, 3) or [64, 64, 64].
            Tile must have same the number of dimensions as data.
            # TODO: it should be possible to create tiles with less dimensions than data

        :param overlap: int, float, tuple, list
            If int, the same overlap in each dimension.
            If float, percentage of a tile_size to use for overlap (from 0.0 to 1.0).
            If tuple or list, size of the overlap in. Must be smaller than tile_shape.
            Default is 0.0.

        :param channel_dimension: int, None
            Used to specify the channel dimension, the dimension that will not be tiled.
            Usually it is the last or the first dimension of the array.
            Default is None, no channel dimension in the data.

        :param mode: str
            Mode defines how the data will be tiled.
            # TODO: allow a user supplied function, Callable

            One of the following string values:
                `constant` (default)
                    If a tile is smaller than `tile_shape`, pad it with a constant value to match `tile_shape`.
                    Set the value with the keyword 'constant_value'.
                'drop'
                    If a tile is smaller than `tile_shape`, ignore it.
                'irregular'
                    Allow tiles to be smaller than `tile_shape`.
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

        :param constant_value: float
            Used in 'constant' mode. The value to set the padded values for each axis.

        """

        # Image and tile shapes
        self.image_shape = np.asarray(image_shape).astype(int)
        self.tile_shape = np.asarray(tile_shape).astype(int)
        self._n_dim = len(image_shape)
        if (self.tile_shape <= 0).any() or (self.image_shape <= 0).any():
            raise ValueError('Tile and data shapes must be tuple or lists of positive numbers.')
        if self.tile_shape.size != self.image_shape.size:
            raise ValueError('Tile and data shapes must have the same length.')

        # Tiling mode
        self.mode = mode
        if self.mode not in self.TILING_MODES:
            raise ValueError(f'{self.mode} is an unsupported tiling mode, please check the documentation.')

        # Constant value used for constant tiling mode
        self.constant_value = constant_value

        # Channel dimension
        self.channel_dimension = channel_dimension
        if self.channel_dimension and \
                ((self.channel_dimension < 0) or (self.channel_dimension >= len(self.image_shape))):
            raise ValueError(f'Specified channel dimension is out of bounds '
                             f'(should be None or an integer from 0 to {len(self.image_shape) - 1}).')

        # Overlap and step
        self.overlap = overlap
        if isinstance(self.overlap, float) and (self.overlap < 0 or self.overlap > 1.0):
            raise ValueError('Overlap, if float, must be in range of 0.0 (0%) to 1.0 (100%).')
        if (isinstance(self.overlap, list) or isinstance(self.overlap, tuple)) \
                and (np.any((self.tile_shape - np.array(self.overlap)) <= 0)):
            raise ValueError('Overlap size much be smaller than tile_shape.')

        if isinstance(self.overlap, list) or isinstance(self.overlap, tuple):
            # overlap is given directly
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

        # Calculate mosaic (collection of tiles) shape
        div, mod = np.divmod([image_shape[d] - self._tile_overlap[d] for d in range(self._n_dim)], self._tile_step)
        if self.mode == 'drop':
            self._indexing_shape = div
        else:
            self._indexing_shape = div + (mod != 0)
        if self.channel_dimension is not None:
            self._indexing_shape[self.channel_dimension] = 1

        # Calculate new shape assuming tiles are padded
        if self.mode == 'irregular':
            self._new_shape = self.image_shape
        else:
            self._new_shape = (self._indexing_shape * self._tile_step) + self._tile_overlap
        self._shape_diff = self._new_shape - self.image_shape
        if self.channel_dimension is not None:
            self._shape_diff[self.channel_dimension] = 0

        # If channel dimension is given, set tile_step of that dimension to 0
        if self.channel_dimension is not None:
            self._tile_step[self.channel_dimension] = 0

        # Tile indexing
        self._tile_index = np.vstack(np.meshgrid(*[np.arange(0, x) for x in self._indexing_shape], indexing='ij'))
        self._tile_index = self._tile_index.reshape(self._n_dim, -1).T
        self.n_tiles = len(self._tile_index)

    def __len__(self) -> int:
        """ Returns number of tiles produced by tiling. """
        return self.n_tiles

    def __repr__(self) -> str:
        return f'Tiler split {list(self.image_shape)} data into {len(self)} tiles of {list(self.tile_shape)}.' \
               f'\n\tMosaic shape: {list(self._indexing_shape)}' \
               f'\n\tTileable shape: {list(self._new_shape)}' \
               f'\n\tTile overlap: {self.overlap}' \
               f'\n\tElement step: {list(self._tile_step)}' \
               f'\n\tMode: {self.mode}' \
               f'\n\tChannel dimension: {self.channel_dimension}'

    def __call__(self, data: np.ndarray, progress_bar: bool = False,
                 batch_size: int = 1, drop_last: bool = False,
                 copy_data: bool = True) -> \
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

        :param copy_data: bool
            If true, returned tile is a copy. Otherwise, it is a view.
            Default is True.

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

        for tile_i in tqdm(range(self.n_tiles), desc='Tiling', disable=not progress_bar, unit='tile'):
            yield tile_i, self.get_tile(data, tile_i, copy=copy_data)

    def get_tile(self, data: Union[np.ndarray, None], tile_id: int, copy: bool = True) -> np.ndarray:
        """
        Returns tile content.

        :param data: np.ndarray
            Data from which tile_id-th tile will be taken.

        :param tile_id: int
            Specify which tile to return. Must be smaller than number of tiles.

        :param copy: bool
            If true, returned tile is a copy. Otherwise, it is a view.
            Default is True.

        :return: np.ndarray
            Content of tile number tile_id. Padded if necessary.
        """

        if (tile_id < 0) or (tile_id >= self.n_tiles):
            raise IndexError(f'Out of bounds, there is no tile {tile_id}.'
                             f'There are {len(self) - 1} tiles, starting from index 0.')

        # get tile data
        tile_corner = self._tile_index[tile_id] * self._tile_step
        sampling = [slice(tile_corner[d], tile_corner[d] + self.tile_shape[d]) for d in range(self._n_dim)]
        tile_data = data[tuple(sampling)]

        if copy:
            tile_data = tile_data.copy()

        shape_diff = self.tile_shape - tile_data.shape
        if (self.mode != 'irregular') and np.any(shape_diff > 0):
            if self.mode == 'constant':
                tile_data = np.pad(tile_data, list((0, diff) for diff in shape_diff), mode=self.mode,
                                   constant_values=self.constant_value)
            elif self.mode == 'reflect' or self.mode == 'edge' or self.mode == 'wrap':
                tile_data = np.pad(tile_data, list((0, diff) for diff in shape_diff), mode=self.mode)

        return tile_data

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
