""" Contains CIFAR datasets """

import os
import tempfile
import logging
import urllib.request
import pickle
import tarfile

import PIL
import tqdm
import numpy as np

from ..dsindex import DatasetIndex
from .base import ImagesOpenset


logger = logging.getLogger('cifar')


class BaseCIFAR(ImagesOpenset):
    """ The base class for the CIFAR dataset """
    SOURCE_URL = None
    LABELS_KEY = None
    TRAIN_NAME_ID = None
    TEST_NAME_ID = None

    def __init__(self, *args, bar=False, preloaded=None, train_test=True, **kwargs):
        self.bar = tqdm.tqdm(total=6) if bar else None
        super().__init__(*args, preloaded=preloaded, train_test=train_test, **kwargs)
        if self.bar:
            self.bar.close()

    def download(self, path=None):
        """ Load data from a web site and extract into numpy arrays """

        def _extract(archive_file, member):
            data = pickle.load(archive_file.extractfile(member), encoding='bytes')
            if self.bar:
                self.bar.update(1)
            return data

        def _gather_extracted(all_res):
            images = np.concatenate([res[b'data'] for res in all_res]).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
            images = np.array([PIL.Image.fromarray(image) for image in images], dtype=object)
            labels = np.concatenate([res[self.LABELS_KEY] for res in all_res])
            return images, labels

        if path is None:
            path = tempfile.gettempdir()
        filename = os.path.basename(self.SOURCE_URL)
        localname = os.path.join(path, filename)

        if not os.path.isfile(localname):
            logger.info("Downloading %s", filename)
            urllib.request.urlretrieve(self.SOURCE_URL, localname)
            logger.info("Downloaded %s", filename)
            if self.bar:
                self.bar.update(1)

        logger.info("Extracting...")
        with tarfile.open(localname, "r:gz") as archive_file:
            files_in_archive = archive_file.getmembers()

            data_files = [one_file for one_file in files_in_archive if self.TRAIN_NAME_ID in one_file.name]
            all_res = [_extract(archive_file, one_file) for one_file in data_files]
            train_data = _gather_extracted(all_res)

            test_files = [one_file for one_file in files_in_archive if self.TEST_NAME_ID in one_file.name]
            all_res = [_extract(archive_file, one_file) for one_file in test_files]
            test_data = _gather_extracted(all_res)
        logger.info("Extracted")

        self._train_index = DatasetIndex(np.arange(len(train_data[0])))
        self._test_index = DatasetIndex(np.arange(len(test_data[0])))
        return train_data, test_data


class CIFAR10(BaseCIFAR):
    """ CIFAR10 dataset

    Examples
    --------
    .. code-block:: python

        # download CIFAR data, split into train/test and create dataset instances
        cifar = CIFAR10()

        # iterate over the dataset
        for batch in cifar.train.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=2):
            # do something with a batch
    """
    SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    LABELS_KEY = b"labels"
    TRAIN_NAME_ID = "data_batch"
    TEST_NAME_ID = "test_batch"
    num_classes = 10


class CIFAR100(BaseCIFAR):
    """ CIFAR100 dataset

    Examples
    --------
    .. code-block:: python

        # download CIFAR data, split into train/test and create dataset instances
        cifar = CIFAR100()

        # iterate over the dataset
        for batch in cifar.train.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=5):
            # do something with a batch
    """
    SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    LABELS_KEY = b"fine_labels"
    TRAIN_NAME_ID = "train"
    TEST_NAME_ID = "test"
    num_classes = 100
