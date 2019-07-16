""" DatasetIndex """
import os
import sys
import math
import glob
from collections.abc import Iterable
import tqdm
import numpy as np

from .base import Baseset


class DatasetIndex(Baseset):
    """ Stores an index for a dataset.
    The index should be 1-d array-like, e.g. numpy array, pandas Series, etc.

    Parameters
    ----------
    index : int, 1-d array-like or callable
        defines structure of DatasetIndex

    Examples
    --------
    >>> index = DatasetIndex(all_item_ids)

    >>> index.split([0.8, 0.2])

    >>> item_pos = index.get_pos(item_id)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos = self.build_pos()
        self._random_state = None

    @classmethod
    def from_index(cls, *args, **kwargs):
        """Create index from another index. """
        return cls(*args, **kwargs)

    @classmethod
    def concat(cls, *index_list):
        """Create index by concatenating other indices.

        Parameters
        ----------
        index_list : list
            Indices to be concatenated. Each item is expected to
            contain index property with 1-d sequence of indices.

        Returns
        -------
        DatasetIndex
            Contains one common index.
        """
        return DatasetIndex(np.concatenate([i.index for i in index_list]))

    @staticmethod
    def build_index(index):
        """ Check index type and structure.

        Parameters
        ----------
        index : int, 1-d array-like or callable
            Defines content of DatasetIndex

            - 1-d array-like
                Content is numpy.array

            - int
                Content is numpy.arange() of given length.

            - callable
                Content is return of given function (should be 1-d array-like).

        Raises
        ------
        TypeError
            If 'index' is not 1-dimensional.

        ValueError
            If 'index' is empty.

        Returns
        -------
        numpy.array
            Index to be stored in class instance.
        """
        if callable(index):
            _index = index()
        else:
            _index = index

        if isinstance(_index, DatasetIndex):
            _index = _index.indices
        elif isinstance(_index, int):
            _index = np.arange(_index)
        else:
            # index should allow for advance indexing (i.e. subsetting)
            _index = np.asarray(_index)

        if np.shape(_index) == ():
            _index = _index.reshape(1)

        if len(_index) == 0:
            raise ValueError("Index cannot be empty")

        if len(_index.shape) > 1:
            raise TypeError("Index should be 1-dimensional")

        return _index

    def build_pos(self):
        """ Create a dictionary with positions in the index. """
        if self.indices is None:
            return dict()
        return dict(zip(self.indices, np.arange(len(self))))

    def get_pos(self, index):
        """ Return position of an item in the index.

        Parameters
        ----------
        index : int, str, slice or Iterable
            Items to return positions of.

            - int, str
                Return position of that item in the DatasetIndex.

            - slice, Iterable
                Return positions of multiple items, specified by argument.

        Returns
        -------
        numpy.array
            Positions of specified items in DatasetIndex.

        Examples
        --------
        Create DatasetIndex that holds index of images and get
        position of one of them

        >>> DatasetIndex(['image_0', 'image_1']).get_pos('image_1')
        """
        if isinstance(index, slice):
            start = self._pos[index.start] if index.start is not None else None
            stop = self._pos[index.stop] if index.stop is not None else None
            pos = slice(start, stop, index.step)
        elif isinstance(index, str):
            pos = self._pos[index]
        elif isinstance(index, Iterable):
            pos = np.asarray([self._pos[ix] for ix in index])
        else:
            pos = self._pos[index]
        return pos

    def subset_by_pos(self, pos):
        """ Return subset of index by given positions in the index.

        Parameters
        ----------
        pos : int, slice, list or numpy.array
            Positions of items to include in subset.

        Returns
        -------
        numpy.array
            Subset of DatasetIndex.index.
        """
        return self.index[pos]

    def create_subset(self, index):
        """ Return a new index object based on the subset of indices given. """
        return type(self)(index)

    def split(self, shares=0.8, shuffle=False):
        """ Split index into train, test and validation subsets.

        Shuffles index if necessary.

        Subsets are available as `.train`, `.test` and `.validation` respectively.

        Parameters
        ----------
        shares : float or tuple of floats
            train, test and validation shares.

        shuffle : bool, int, class:`numpy.random.RandomState` or callable
            specifies the order of items, could be:

            - bool - if `False`, items go sequentionally, one after another as they appear in the index.
                if `True`, items are shuffled randomly before each epoch.

            - int - a seed number for a random shuffle.

            - :class:`numpy.random.RandomState` instance.

            - callable - a function which takes an array of item indices in the initial order
                (as they appear in the index) and returns the order of items.

        Notes
        -----
        If tuple of 3 floats is passed, then validation subset is always present.

        Examples
        ---------

        split into train / test in 80/20 ratio

        >>> index.split()

        split into train / test / validation in 60/30/10 ratio

        >>> index.split([0.6, 0.3])

        split into train / test / validation in 50/30/20 ratio

        >>> index.split([0.5, 0.3, 0.2])

        use 1 sample as validation and split the rest evenly to train / test

        >>> index.split([0.5, 0.5, 0])
        """
        train_share, test_share, valid_share = self.calc_split(shares)

        if shuffle:
            order = self.shuffle(shuffle)
        else:
            order = np.arange(len(self))

        if valid_share > 0:
            validation_pos = order[:valid_share]
            self.validation = self.create_subset(self.subset_by_pos(validation_pos))
        if test_share > 0:
            test_pos = order[valid_share : valid_share + test_share]
            self.test = self.create_subset(self.subset_by_pos(test_pos))
        if train_share > 0:
            train_pos = order[valid_share + test_share:]
            self.train = self.create_subset(self.subset_by_pos(train_pos))

    def shuffle(self, shuffle, iter_params=None):
        """ Permute indices

        Parameters
        ----------
        shuffle : bool, int, class:`numpy.random.RandomState` or callable
            specifies the order of items, could be:

            - bool - if `False`, items go sequentionally, one after another as they appear in the index.
                if `True`, items are shuffled randomly before each epoch.

            - int - a seed number for a random shuffle.

            - :class:`numpy.random.RandomState` instance.

            - callable - a function which takes an array of item indices in the initial order
                (as they appear in the index) and returns the order of items.

        Returns
        -------
        ndarray
            a permuted order for indices
        """
        if iter_params is None:
            iter_params = self.get_default_iter_params()

        if iter_params['_order'] is None:
            order = np.arange(len(self))
        else:
            order = iter_params['_order']

        if isinstance(shuffle, bool):
            if shuffle:
                order = np.random.permutation(order)
        elif isinstance(shuffle, int):
            if iter_params['_random_state'] is None or iter_params['_random_state'].seed != shuffle:
                iter_params['_random_state'] = np.random.RandomState(shuffle)
            order = iter_params['_random_state'].permutation(order)
        elif isinstance(shuffle, np.random.RandomState):
            if iter_params['_random_state'] != shuffle:
                iter_params['_random_state'] = shuffle
            order = iter_params['_random_state'].permutation(order)
        elif callable(shuffle):
            order = shuffle(self.indices)
        else:
            raise ValueError("shuffle could be bool, int, numpy.random.RandomState or callable")
        return order

    def next_batch(self, batch_size, shuffle=False, n_iters=None, n_epochs=None, drop_last=False, iter_params=None):
        """ Return the next batch

        Parameters
        ----------
        batch_size : int
            Desired number of items in the batch (the actual batch could contain fewer items)

        shuffle : bool, int, class:`numpy.random.RandomState` or callable
            Specifies the order of items, could be:

            - bool
                If `False`, items go sequentionally, one after another as they appear in the index.
                If `True`, items are shuffled randomly before each epoch.

            - int
                A seed number for a random shuffle.

            - :class:`numpy.random.RandomState` instance
                Class for a reproducible random shuffle.

            - callable
                A function which takes an array of item indices in the initial order
                (as they appear in the index) and returns the order of items.

        n_iters : int
            Number of iterations to make (only one of `n_iters` and `n_epochs` should be specified).

        n_epochs : int
            Number of epochs required (only one of `n_iters` and `n_epochs` should be specified).

        drop_last : bool
            If `True`, drops the last batch (in each epoch) if it contains fewer than `batch_size` items.
            If `False`, than the last batch in each epoch could contain repeating indices (which might be a problem)
            and the very last batch could contain fewer than `batch_size` items.

            For instance, `next_batch(3, shuffle=False, n_epochs=2, drop_last=False)` for a dataset with 4 items returns
            indices [0,1,2], [3,0,1], [2,3].
            While `next_batch(3, shuffle=False, n_epochs=2, drop_last=True)` returns indices [0,1,2], [0,1,2].

            Take into account that `next_batch(3, shuffle=True, n_epochs=2, drop_last=False)` could return batches
            [3,0,1], [2,0,2], [1,3]. Here the second batch contains two items with the same index "2".
            This might become a problem if some action uses `batch.get_pos()` or `batch.index.get_pos()` methods so that
            one of the identical items will be missed.
            However, there is nothing to worry about if you don't iterate over batch items explicitly
            (i.e. `for item in batch`) or implicitly (through `batch[ix]`).

        Raises
        ------
        StopIteration
            When `n_epochs` has been reached and there is no batches left in the dataset.

        ValueError
            When `n_epochs` and `n_iters` have been passed at the same time.

        Examples
        --------

        ::

            for i in range(MAX_ITER):
                index_batch = index.next_batch(BATCH_SIZE, shuffle=True, n_epochs=2, drop_last=True):
                # do whatever you want
        """
        if n_iters is not None and n_epochs is not None:
            raise ValueError("Only one of n_iters and n_epochs should be specified.")

        if iter_params is None:
            iter_params = self._iter_params

        # The previous iteration was the last one to perform, so stop iterating
        if iter_params['_stop_iter']:
            if 'bar' in iter_params:
                iter_params['bar'].close()
            raise StopIteration("Dataset is over. No more batches left.")

        if iter_params['_order'] is None:
            iter_params['_order'] = self.shuffle(shuffle, iter_params)
        num_items = len(iter_params['_order'])

        rest_items = None
        if iter_params['_start_index'] + batch_size >= num_items:
            rest_items = np.copy(iter_params['_order'][iter_params['_start_index']:])
            rest_of_batch = iter_params['_start_index'] + batch_size - num_items
            if rest_of_batch > 0:
                if drop_last:
                    rest_items = None
                    rest_of_batch = batch_size
            iter_params['_start_index'] = 0
            iter_params['_n_epochs'] += 1
            iter_params['_order'] = self.shuffle(shuffle, iter_params)
        else:
            rest_of_batch = batch_size

        new_items = iter_params['_order'][iter_params['_start_index'] : iter_params['_start_index'] + rest_of_batch]
        if rest_items is None:
            batch_items = new_items
        else:
            batch_items = np.concatenate((rest_items, new_items))

        if n_iters is not None and iter_params['_n_iters'] >= n_iters or \
           n_epochs is not None and iter_params['_n_epochs'] >= n_epochs:
            if 'bar' in iter_params:
                iter_params['bar'].close()
            if n_iters is not None or drop_last and (rest_items is None or len(rest_items) < batch_size):
                raise StopIteration("Dataset is over. No more batches left.")
            iter_params['_stop_iter'] = True
            return self.create_batch(rest_items, pos=True)

        iter_params['_n_iters'] += 1
        iter_params['_start_index'] += rest_of_batch
        return self.create_batch(batch_items, pos=True)


    def gen_batch(self, batch_size, shuffle=False, n_iters=None, n_epochs=None, drop_last=False, bar=False):
        """ Generate batches

        Parameters
        ----------
        batch_size : int
            Desired number of items in the batch (the actual batch could contain fewer items).

        shuffle : bool, int, class:`numpy.random.RandomState` or callable
            Specifies the order of items, could be:

            - bool
                If `False`, items go sequentionally, one after another as they appear in the index.
                If `True`, items are shuffled randomly before each epoch.

            - int
                A seed number for a random shuffle.

            - :class:`numpy.random.RandomState` instance
                Class for a reproducible random shuffle.

            - callable
                A function which takes an array of item indices in the initial order
                (as they appear in the index) and returns the order of items.

        n_iters : int
            Number of iterations to make (only one of `n_iters` and `n_epochs` should be specified).

        n_epochs : int
            Number of epochs required (only one of `n_iters` and `n_epochs` should be specified).

        drop_last : bool
            If `True`, drops the last batch (in each epoch) if it contains fewer than `batch_size` items.
            If `False`, than the last batch in each epoch could contain repeating indices (which might be a problem)
            and the very last batch could contain fewer than `batch_size` items.

            For instance, `gen_batch(3, shuffle=False, n_epochs=2, drop_last=False)` for a dataset with 4 items returns
            indices [0,1,2], [3,0,1], [2,3].
            While `gen_batch(3, shuffle=False, n_epochs=2, drop_last=True)` returns indices [0,1,2], [0,1,2].

            Take into account that `gen_batch(3, shuffle=True, n_epochs=2, drop_last=False)` could return batches
            [3,0,1], [2,0,2], [1,3]. Here the second batch contains two items with the same index "2".
            This might become a problem if some action uses `batch.get_pos()` or `batch.index.get_pos()` methods so that
            one of the identical items will be missed.
            However, there is nothing to worry about if you don't iterate over batch items explicitly
            (i.e. `for item in batch`) or implicitly (through `batch[ix]`).

        bar : bool, 'n' or callable
            Whether to show a progress bar.
            If 'n', then uses `tqdm_notebook`. If callable, it must have the same signature as `tqdm`.

        Yields
        ------
        An instance of the same class with a subset of indices

        Raises
        ------
        ValueError
            When `n_epochs` and `n_iters` have been passed at the same time.

        Examples
        --------

        ::

            for index_batch in index.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=2, drop_last=True):
                # do whatever you want
        """
        iter_params = self.get_default_iter_params()
        if bar:
            if n_iters is not None:
                total = n_iters
            elif n_epochs is None:
                total = sys.maxsize
            elif drop_last:
                total = len(self) // batch_size * n_epochs
            else:
                total = math.ceil(len(self) * n_epochs / batch_size)

            if callable(bar):
                iter_params['bar'] = bar(total=total)
            elif bar == 'n':
                iter_params['bar'] = tqdm.tqdm_notebook(total=total)
            else:
                iter_params['bar'] = tqdm.tqdm(total=total)

        while True:
            if n_epochs is not None and iter_params['_n_epochs'] >= n_epochs:
                return
            try:
                batch = self.next_batch(batch_size, shuffle, n_iters, n_epochs, drop_last, iter_params)
            except StopIteration:
                return
            if 'bar' in iter_params:
                iter_params['bar'].update(1)
            yield batch


    def create_batch(self, index, pos=True, as_array=False, *args, **kwargs):
        """ Create a batch from given indices.

        Parameters
        ----------
        index : int, slice, list, numpy.array or DatasetIndex
            If 'pos' is True, then 'index' should contain
            positions of items in the current index to be returned as
            separate batch.

            If 'pos' is False, then 'index' should contain
            indices to be returned as separate batch
            (so expected batch is just the very same index).

        pos : bool
            Whether to return indices or positions

        as_array : bool
            Whether to return array or an instance of DatasetIndex

        Returns
        -------
        DatasetIndex or numpy.array
            Part of initial DatasetIndex, specified by 'index'.

        Examples
        --------
        Create DatasetIndex with first 100 natural numbers, then
        get batch with every second item

        >>> DatasetIndex(100).create_batch(index=2*numpy.arange(50))
        """
        _ = args, kwargs
        if isinstance(index, DatasetIndex):
            _index = index.indices
        else:
            _index = index
        if pos:
            batch = self.subset_by_pos(_index)
        else:
            batch = _index
        if not as_array:
            batch = self.create_subset(batch)
        return batch


class FilesIndex(DatasetIndex):
    """ Index with the list of files or directories with the given path pattern

    Examples
    --------

    Create a sorted index of files in a directory:

    >>> fi = FilesIndex(path='/path/to/data/files/*', sort=True)

    Create an index of directories through all subdirectories:

    >>> fi = FilesIndex(path='/path/to/data/archive*/patient*', dirs=True)

    Create an index of files in several directories, and file extensions are ignored:

    >>> fi = FilesIndex(path=['/path/to/archive/2016/*','/path/to/current/file/*'], no_ext=True)

    To get a path to the file call `get_fullpath(index_id)`:

    >>> path = fi.get_fullpath(some_id)

    Split into train / test / validation in 80/15/5 ratio

    >>> fi.split([0.8, 0.15])

    Get a position of a customer in the index

    >>> item_pos = fi.get_pos(customer_id)

    """
    def __init__(self, *args, **kwargs):
        self._paths = None
        self.dirs = False
        super().__init__(*args, **kwargs)

    def build_index(self, index=None, path=None, *args, **kwargs):
        """ Build index from a path string or an index given. """
        if path is None:
            return self.build_from_index(index, *args, **kwargs)
        return self.build_from_path(path, *args, **kwargs)

    def build_from_index(self, index, paths, dirs):
        """ Build index from another index for indices given. """
        if isinstance(index, DatasetIndex):
            index = index.indices
        if isinstance(paths, dict):
            self._paths = dict((file, paths[file]) for file in index)
        else:
            self._paths = dict((file, paths[pos]) for pos, file in np.ndenumerate(index))
        self.dirs = dirs
        return index

    def build_from_path(self, path, dirs=False, no_ext=False, sort=False):
        """ Build index from a path/glob or a sequence of paths/globs. """
        if isinstance(path, str):
            paths = [path]
        else:
            paths = path

        _all_index = None
        _all_paths = dict()
        for one_path in paths:
            _index, _paths = self.build_from_one_path(one_path, dirs, no_ext)
            if _all_index is None:
                _all_index = _index
            else:
                _all_index = np.concatenate((_all_index, _index))
            _all_paths.update(_paths)

        if sort:
            _all_index.sort()
        self._paths = _all_paths
        self.dirs = dirs

        return _all_index

    def build_from_one_path(self, path, dirs=False, no_ext=False):
        """ Build index from a path/glob. """
        check_fn = os.path.isdir if dirs else os.path.isfile
        pathlist = glob.iglob(path, recursive=True)
        _full_index = np.asarray([self.build_key(fname, no_ext) for fname in pathlist if check_fn(fname)])
        if len(_full_index):
            _index = _full_index[:, 0]
            _paths = _full_index[:, 1]
        else:
            _index, _paths = [], []
        _paths = dict(zip(_index, _paths))
        return _index, _paths

    @staticmethod
    def build_key(fullpathname, no_ext=False):
        """ Create index item from full path name. """
        key_name = os.path.basename(fullpathname)
        if no_ext:
            dot_position = key_name.rfind('.')
            dot_position = dot_position if dot_position > 0 else len(key_name)
            key_name = key_name[:dot_position]

        return key_name, fullpathname

    def get_fullpath(self, key):
        """ Return the full path name for an item in the index. """
        return self._paths[key]

    def create_subset(self, index):
        """ Return a new FilesIndex based on the subset of indices given. """
        return type(self).from_index(index=index, paths=self._paths, dirs=self.dirs)
