""" Contains the base class for open datasets """

from .. import Dataset
from .. import ImagesBatch


class Openset(Dataset):
    """ The base class for open datasets """
    def __init__(self, index=None, batch_class=None, train_test=False, path=None, preloaded=None):
        self.train_test = train_test
        self._train_index, self._test_index = None, None
        if preloaded is None:
            preloaded = self.download(path=path)
        super().__init__(index, batch_class, preloaded=preloaded)
        if train_test:
            self.split()

    @staticmethod
    def uild_index(index):
        """ Create an index """
        if index is not None:
            return super().build_index(index)
        return None

    def download(self, path):
        """ Download a dataset from the source web-site """
        _ = path
        return None

    def split(self, shares=0.8, shuffle=False):
        if self.train_test:
            train_data, test_data = self.preloaded  # pylint:disable=unpacking-non-sequence
            self.train = type(self)(self._train_index, self.batch_class, train_test=False, preloaded=train_data)
            self.test = type(self)(self._test_index, self.batch_class, train_test=False, preloaded=test_data)
        else:
            super().split(shares, shuffle)


class ImagesOpenset(Openset):
    """ The base class for open datasets with images """
    def __init__(self, index=None, batch_class=ImagesBatch, *args, **kwargs):
        super().__init__(index, batch_class, *args, **kwargs)
