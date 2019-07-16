""" Base class """
import numpy as np


class Baseset:
    """ Base class """
    def __init__(self, *args, **kwargs):
        self._index = self.build_index(*args, **kwargs)

        self.train = None
        self.test = None
        self.validation = None

        self._iter_params = None
        self.reset_iter()

    @staticmethod
    def build_index(index, *args, **kwargs):
        """ Create the index. Child classes should generate index from the arguments given """
        _ = args, kwargs
        return index

    @property
    def index(self):
        """:class:`dataset.DatasetIndex` : the dataset's index """
        return self._index

    @property
    def indices(self):
        """:class:`numpy.ndarray` : an array with the indices """
        if isinstance(self.index, Baseset):
            return self.index.indices
        return self.index

    def __len__(self):
        if self.indices is None:
            return 0
        return len(self.indices)

    @property
    def size(self):
        """: int - number of items in the set """
        return len(self)

    @property
    def is_split(self):
        """bool : True if dataset has been split into train / test / validation subsets """
        return self.train is not None or self.test is not None or self.validation is not None

    def calc_split(self, shares=0.8):
        """ Calculate split into train, test and validation subsets

        Parameters
        ----------
        shares : float or a sequence of floats
            A share of train, test and validation subset respectively.

        Returns
        -------
        a tuple which contains number of items in train, test and validation subsets

        Raises
        ------
        ValueError
            * if shares has more than 3 items
            * if sum of shares is greater than 1
            * if this set does not have enough items to split

        Examples
        --------
        Split into train / test in 80/20 ratio

        >>> some_set.calc_split()

        Split into train / test / validation in 60/30/10 ratio

        >>> some_set.calc_split([0.6, 0.3])

        Split into train / test / validation in 50/30/20 ratio

        >>> some_set.calc_split([0.5, 0.3, 0.2])
        """
        _shares = [shares] if isinstance(shares, (int, float)) else shares
        _shares = _shares if len(_shares) > 2 else _shares + [.0]
        _shares = np.array(_shares).ravel()         # pylint: disable=no-member
        n_items = len(self)

        if _shares.shape[0] > 3:
            raise ValueError("Shares must have no more than 3 elements")
        if _shares.sum() > 1:
            raise ValueError("Shares must sum to 1:", shares)
        if n_items < len(_shares):
            raise ValueError("A set of size %d cannot be split into %d subsets" % (n_items, len(_shares)))

        _shares[-1] = 1 - _shares[:-1].sum()
        if _shares[-1] == 0:
            _shares = _shares[:-1]
        _lens = np.round(_shares * n_items).astype('int')

        for s, _ in enumerate(_shares):
            _lens[s] = _lens[s] if _shares[s] > 0 and _lens[s] >= 1 else 1
        _lens = np.pad(_lens, (0, 3 - len(_lens)), 'constant')

        train_len, test_len, valid_len = _lens
        train_len = max(0, n_items - test_len - valid_len)

        return train_len, test_len, valid_len


    def create_subset(self, index):
        """ Create a new subset based on the given index subset """
        raise NotImplementedError("create_subset should be defined in child classes")


    def split(self, shares=0.8, shuffle=False):
        """ Split the dataset into train, test and validation sub-datasets.
        Subsets are available as `.train`, `.test` and `.validation` respectively.

        Parameters
        ----------
        shares : float, tuple of 2 floats, or tuple of 3 floats
            train/test/validation shares. Default is 0.8.

        shuffle : bool, :class:`numpy.random.RandomState`, int or callable
            whether to randomize items order before splitting into subsets. Default is `False`. Can be

            * `bool` : `False` - to make subsets in the order of indices in the index,
                       `True` - to make random subsets.
            * a :class:`numpy.random.RandomState` object which has an inplace shuffle method.
            * `int` - a random seed number which will be used internally to create
                      a :class:`numpy.random.RandomState` object.
            * callable - a function which gets an order and returns a shuffled order.

        Examples
        --------
        Split into train / test in 80/20 ratio

        >>> dataset.split()

        Split into train / test / validation in 60/30/10 ratio

        >>> dataset.split([0.6, 0.3])

        Split into train / test / validation in 50/30/20 ratio

        >>> dataset.split([0.5, 0.3, 0.2])
        """
        self.index.split(shares, shuffle)

        if self.index.train is not None:
            self.train = self.create_subset(self.index.train)
        if self.index.test is not None:
            self.test = self.create_subset(self.index.test)
        if self.index.validation is not None:
            self.validation = self.create_subset(self.index.validation)

    def get_default_iter_params(self):
        """ Return iteration params with default values to start iteration from scratch """
        return dict(_stop_iter=False, _start_index=0, _order=None, _n_iters=0, _n_epochs=0, _random_state=None)

    def reset_iter(self):
        """ Clear all iteration metadata in order to start iterating from scratch """
        self._iter_params = self.get_default_iter_params()
        if isinstance(self.index, Baseset):
            self.index.reset_iter()

    def gen_batch(self, batch_size, shuffle=False, n_iters=None, n_epochs=None, drop_last=False, bar=False,
                  *args, **kwargs):
        """ Generate batches """
        for ix_batch in self.index.gen_batch(batch_size, shuffle, n_iters, n_epochs, drop_last, bar):
            batch = self.create_batch(ix_batch, *args, **kwargs)
            yield batch

    def next_batch(self, batch_size, shuffle=False, n_iters=None, n_epochs=None, drop_last=False,
                   iter_params=None, *args, **kwargs):
        """ Return a batch """
        batch_index = self.index.next_batch(batch_size, shuffle, n_iters, n_epochs, drop_last, iter_params)
        batch = self.create_batch(batch_index, *args, **kwargs)
        return batch

    def create_batch(self, batch_indices, pos=True):
        """ Create batch with indices given """
        raise NotImplementedError("create_batch should be implemented in child classes")
