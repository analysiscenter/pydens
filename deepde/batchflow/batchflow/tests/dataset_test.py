# pylint: disable=missing-docstring, redefined-outer-name

import pytest
import numpy as np

from batchflow import Dataset, Batch, ImagesBatch, DatasetIndex, Pipeline


@pytest.fixture
def dataset():
    index = DatasetIndex(100)
    return Dataset(index, Batch)


class TestDataset:
    def test_from_dataset(self, dataset):
        new_index = DatasetIndex(25)
        new_ds = dataset.from_dataset(dataset, new_index)
        assert isinstance(new_ds, dataset.__class__)
        assert new_ds.batch_class == dataset.batch_class
        assert (new_ds.index.index == new_index.index).all()

    def test_from_dataset_new_batch_class(self, dataset):
        new_index = DatasetIndex(25)
        new_ds = Dataset.from_dataset(dataset, new_index, batch_class=ImagesBatch)
        assert isinstance(new_ds, dataset.__class__)
        assert new_ds.batch_class == ImagesBatch

    def test_build_index(self):
        new_index = Dataset.build_index(25)
        assert isinstance(new_index, DatasetIndex)

    def test_create_subset(self, dataset):
        new_index = DatasetIndex(25)
        new_ds = dataset.create_subset(new_index)
        assert isinstance(new_ds, dataset.__class__)
        assert np.isin(new_ds.indices, dataset.indices).all()

    def test_create_subset_wrong_index(self, dataset):
        wrong_index = DatasetIndex(np.arange(200, 225))
        with pytest.raises(IndexError) as error:
            dataset.create_subset(wrong_index)
            assert 'IndexError' in str(error)

    def test_create_batch(self, dataset):
        target_index = DatasetIndex(5)
        new_batch = dataset.create_batch(target_index)
        assert isinstance(new_batch, dataset.batch_class)
        assert len(new_batch.indices) == len(target_index.indices)

    def test_pipeline(self, dataset):
        pipeline_config = {}
        new_pipeline = dataset.pipeline(pipeline_config)
        assert isinstance(new_pipeline, Pipeline)

    def test_rshift(self, dataset):
        pipeline_config = {}
        new_pipeline = dataset.pipeline(pipeline_config)
        train_pipeline = (new_pipeline << dataset)
        assert isinstance(train_pipeline, Pipeline)

    def test_split_no_validation(self, dataset):
        assert dataset.train is None
        train_part, test_part = 0.8, 0.2
        dataset.split([train_part, test_part])
        assert dataset.train is not None
        assert dataset.test is not None
        assert dataset.validation is None
        assert len(dataset.train) == 80
        assert len(dataset.test) == 20

    def test_split_with_validation(self, dataset):
        train_part, test_part, validation_part = 0.7, 0.2, 0.1
        dataset.split([train_part, test_part, validation_part])
        assert dataset.validation is not None
        assert len(dataset.train) == 70
        assert len(dataset.test) == 20
        assert len(dataset.validation) == 10

    def test_split_with_validation_implicit(self, dataset):
        train_part, test_part, _ = 0.6, 0.25, 0.15
        dataset.split([train_part, test_part])
        assert dataset.validation is not None
        assert len(dataset.train) == 60
        assert len(dataset.test) == 25
        assert len(dataset.validation) == 15
