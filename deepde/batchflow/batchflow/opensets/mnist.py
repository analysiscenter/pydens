""" Contains MNIST dataset """
import os
import logging
import tempfile
import urllib.request
import gzip

import PIL
import tqdm
import numpy as np


from . import ImagesOpenset
from .. import DatasetIndex
from .. import parallel, any_action_failed


logger = logging.getLogger('mnist')


class MNIST(ImagesOpenset):
    """ MNIST dataset

    Examples
    --------

    ::

        # download MNIST data, split into train/test and create dataset instances
        mnist = MNIST()
        # iterate over the dataset
        for batch in mnist.train.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=2):
            # do something with a batch


        # download MNIST data and show progress bar
        mnist = MNIST(bar=True)
    """

    TRAIN_IMAGES_URL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    TRAIN_LABELS_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    TEST_IMAGES_URL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    TEST_LABELS_URL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    ALL_URLS = [TRAIN_IMAGES_URL, TRAIN_LABELS_URL, TEST_IMAGES_URL, TEST_LABELS_URL]
    num_classes = 10

    def __init__(self, *args, bar=False, preloaded=None, train_test=True, **kwargs):
        self.bar = tqdm.tqdm(total=8) if bar else None
        super().__init__(*args, preloaded=preloaded, train_test=train_test, **kwargs)
        if self.bar:
            self.bar.close()

    @property
    def _get_from_urls(self):
        """ List of URLs and type of content (0 - images, 1 - labels) """
        return [[self.ALL_URLS[i], i % 2] for i in range(len(self.ALL_URLS))]

    def _gather_data(self, all_res, *args, **kwargs):
        _ = args, kwargs
        if any_action_failed(all_res):
            raise IOError('Could not download files:', all_res)

        train_data = all_res[0], all_res[1]
        test_data = all_res[2], all_res[3]
        self._train_index = DatasetIndex(len(train_data[0]))
        self._test_index = DatasetIndex(len(test_data[0]))

        return train_data, test_data

    @parallel(init='_get_from_urls', post='_gather_data', target='t')
    def download(self, url, content, path=None):
        """ Load data from the web site """
        logger.info('Downloading %s', url)
        if path is None:
            path = tempfile.gettempdir()
        filename = os.path.basename(url)
        localname = os.path.join(path, filename)
        if not os.path.isfile(localname):
            urllib.request.urlretrieve(url, localname)
            logger.info("Downloaded %s", filename)
        if self.bar:
            self.bar.update(1)

        with open(localname, 'rb') as f:
            data = self._extract_images(f) if content == 0 else self._extract_labels(f)
            if self.bar:
                self.bar.update(1)
        return data

    #
    #  _read32, extract_images, extract_labels are taken from tensorflow
    #
    @staticmethod
    def _read32(bytestream):
        dtype = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dtype)[0]

    def _extract_images(self, f):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
        Args:
          f: A file object that can be passed into a gzip reader.
        Returns:
          data: A 4D uint8 numpy array [index, y, x, depth].
        Raises:
          ValueError: If the bytestream does not start with 2051.
        """
        logger.info('Extracting %s', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)
            return np.array([PIL.Image.fromarray(image) for image in data], dtype=object)

    def _extract_labels(self, f):
        """Extract the labels into a 1D uint8 numpy array [index].
        Args:
          f: A file object that can be passed into a gzip reader.
        Returns:
          labels: a 1D uint8 numpy array.
        Raises:
          ValueError: If the bystream doesn't start with 2049.
        """
        logger.info('Extracting %s', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels
