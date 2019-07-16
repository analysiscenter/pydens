"""Tests for DatasetIndex class.
If possible, methods are tested against DatasetIndex with length of 5.
When random values are needed, 'random_seed' is fixed to be 13.
"""
# pylint: disable=missing-docstring
# pylint: disable=protected-access
import pytest
import numpy as np

from batchflow import DatasetIndex


def test_len():
    dsi = DatasetIndex(5)
    assert len(dsi) == 5


def test_calc_split_raise():
    dsi = DatasetIndex(5)
    with pytest.raises(ValueError):
        dsi.calc_split(shares=[0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        dsi.calc_split(shares=[0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        DatasetIndex(2).calc_split(shares=[0.5, 0.5, 0.5])

def test_calc_split_correctness_1():
    dsi = DatasetIndex(5)
    assert sum(dsi.calc_split()) == 5

def test_calc_split_correctness_2():
    """ If 'shares' contains 2 elements, validation subset is empty. """
    dsi = DatasetIndex(5)
    left = dsi.calc_split(shares=[0.4, 0.6])
    right = (2, 3, 0)
    assert left == right

def test_calc_split_correctness_3():
    """ If 'shares' contains 3 elements, then validation subset is non-empty. """
    dsi = DatasetIndex(5)
    _, _, valid_share = dsi.calc_split(shares=[0.5, 0.5, 0])
    assert valid_share == 0


@pytest.mark.parametrize('constructor', [5,
                                         range(10, 20, 2),
                                         DatasetIndex(5),
                                         ['a', 'b', 'c', 'd', 'e'],
                                         (lambda: np.arange(5)[::-1])])
def test_build_index(constructor):
    """ True content of 'dsi.index' is recovered from the 'constructor'. """
    dsi = DatasetIndex(constructor)
    if isinstance(constructor, int):
        constructor = np.arange(constructor)
    elif isinstance(constructor, DatasetIndex):
        constructor = constructor.index
    elif callable(constructor):
        constructor = constructor()
    assert (dsi.index == constructor).all()

def test_build_index_empty():
    with pytest.raises(ValueError):
        DatasetIndex([])

def test_build_index_multidimensional():
    with pytest.raises(TypeError):
        DatasetIndex([[1], [2]])


def test_get_pos_int():
    dsi = DatasetIndex(5)
    assert dsi.get_pos(4) == 4

def test_get_pos_slice():
    dsi = DatasetIndex(5)
    assert dsi.get_pos(slice(0, 4, 2)) == slice(0, 4, 2)

def test_get_pos_str():
    dsi = DatasetIndex(['a', 'b', 'c', 'd', 'e'])
    assert dsi.get_pos('c') == 2

def test_get_pos_iterable():
    dsi = DatasetIndex(5)
    assert (dsi.get_pos(np.arange(5)) == np.arange(5)).all()


def test_shuffle_bool_false():
    dsi = DatasetIndex(5)
    left = dsi.shuffle(shuffle=False)
    right = np.arange(5)
    assert (left == right).all()

def test_shuffle_bool_true():
    dsi = DatasetIndex(5)
    left = dsi.shuffle(shuffle=True)
    right = np.arange(5)
    assert (left != right).any()
    assert set(left) == set(right)

def test_shuffle_bool_int():
    dsi = DatasetIndex(5)
    left = dsi.shuffle(shuffle=13)
    right = np.arange(5)
    assert (left != right).any()
    assert set(left) == set(right)

def test_shuffle_bool_randomstate():
    dsi = DatasetIndex(5)
    left = dsi.shuffle(shuffle=np.random.RandomState(13))
    right = np.arange(5)
    assert (left != right).any()
    assert set(left) == set(right)

def test_shuffle_bool_cross():
    dsi = DatasetIndex(5)
    left = dsi.shuffle(shuffle=np.random.RandomState(13))
    right = dsi.shuffle(shuffle=13)
    assert (left == right).all()

def test_shuffle_bool_callable():
    """ Callable 'shuffle' should return order. """
    dsi = DatasetIndex(5)
    left = dsi.shuffle(shuffle=(lambda _: np.arange(5)))
    assert (left == np.arange(5)).all()


def test_split_correctness():
    """ Each element of 'index' is used.
    Constants in 'shares' are such that test does not raise errors.
    """
    dsi = DatasetIndex(5)
    shares = .3 - np.random.random(3) *.05
    dsi.split(shares=shares)

    assert set(dsi.index) == (set(dsi.train.index)
                              | set(dsi.test.index)
                              | set(dsi.validation.index))


def test_create_batch_pos_true_list():
    """ When 'pos' is True, method creates new batch by specified positions. """
    dsi = DatasetIndex(range(10, 20, 2))
    left = dsi.create_batch(range(5), pos=True).index
    assert (left == range(10, 20, 2)).all()

def test_create_batch_pos_true_str():
    """ When 'pos' is True, method creates new batch by specified positions. """
    dsi = DatasetIndex(['a', 'b', 'c', 'd', 'e'])
    left = dsi.create_batch(range(5), pos=True).index
    assert (left == ['a', 'b', 'c', 'd', 'e']).all()

def test_create_batch_pos_false_str():
    """ When 'pos' is False, method returns the same, as its first argument. """
    dsi = DatasetIndex(['a', 'b', 'c', 'd', 'e'])
    left = dsi.create_batch(['a', 'e'], pos=False).index
    assert (left == ['a', 'e']).all()

def test_create_batch_pos_false_int():
    """ When 'pos' is False, method returns the same, as its first argument. """
    dsi = DatasetIndex(5)
    left = dsi.create_batch(range(3), pos=False).index
    assert (left == range(3)).all()

def test_create_batch_child():
    """ Method 'create_batch' must be type-preserving. """
    class ChildSet(DatasetIndex):
        # pylint: disable=too-few-public-methods
        pass
    dsi = ChildSet(5)
    assert isinstance(dsi.create_batch(range(5)), ChildSet)


def test_next_batch_stopiter_raise():
    """ Iteration is blocked after end of DatasetIndex. """
    dsi = DatasetIndex(5)
    dsi.next_batch(5, n_epochs=1)
    with pytest.raises(StopIteration):
        dsi.next_batch(5, n_epochs=1)

def test_next_batch_stopiter_pass():
    """ When 'n_epochs' is None it is possible to iterate infinitely. """
    dsi = DatasetIndex(5)
    for _ in range(10):
        dsi.next_batch(1, n_epochs=None)

def test_next_batch_drop_last_false_1():
    """ When 'drop_last' is False 'next_batch' should cycle through index. """
    dsi = DatasetIndex(5)
    left = []
    right = list(np.concatenate([dsi.index, dsi.index]))
    for length in [3, 3, 4]:
        batch = dsi.next_batch(batch_size=length,
                               n_epochs=2,
                               drop_last=False)
        left.extend(list(batch.index))
    assert left == right

def test_next_batch_drop_last_false_2():
    """ When 'drop_last' is False last batch of last epoch can have smaller length. """
    dsi = DatasetIndex(5)
    left = []
    right = [2]*7 + [1] # first seven batches have length of 2, last contains one item
    for _ in range(8):
        batch = dsi.next_batch(batch_size=2,
                               n_epochs=3,
                               drop_last=False)
        left.append(len(batch))
    assert left == right

def test_next_batch_drop_last_true():
    """ Order and contents of generated batches is same at every epoch.
    'shuffle' is False, so dropped indices are always the same.
    """
    dsi = DatasetIndex(5)
    for _ in range(10):
        batch_1 = dsi.next_batch(batch_size=2,
                                 n_epochs=None,
                                 drop_last=True,
                                 shuffle=False)
        batch_2 = dsi.next_batch(batch_size=2,
                                 n_epochs=None,
                                 drop_last=True,
                                 shuffle=False)
        assert (batch_1.index == dsi.index[:2]).all()
        assert (batch_2.index == dsi.index[2:4]).all()


def test_next_batch_smaller():
    """ 'batch_size' is twice as small as length DatasetIndex. """
    dsi = DatasetIndex(5)
    for _ in range(10):
        batch = dsi.next_batch(batch_size=2,
                               n_epochs=None,
                               drop_last=True)
        assert len(batch) == 2

def test_next_batch_bigger():
    """ When 'batch_size' is bigger than length of DatasetIndex, the
    behavior is unstable.
    """
    dsi = DatasetIndex(5)
    with pytest.raises(AssertionError):
        for _ in range(10):
            batch = dsi.next_batch(batch_size=7,
                                   n_epochs=None,
                                   drop_last=True)
            assert len(batch) == 7
