# pylint: disable=redefined-outer-name, missing-docstring, bad-continuation
import sys
import pytest

sys.path.append('..')
from batchflow import B, C, D, F, V, L, R, P, Dataset


@pytest.mark.parametrize('named_expr', [
    C('option'),
    B('size'),
    D('size'),
    V('var'),
    R('normal', 0, 1),
    P('normal', 0, 1),
    F(lambda batch: 0),
    L(lambda: 0),
])
def test_general_get(named_expr):
    pipeline = (Dataset(10).pipeline({'option': 0})
        .init_variable('var')
        .do_nothing(named_expr)
        .run(2, lazy=True)
    )

    failed = False
    try:
        _ = pipeline.next_batch()
    except KeyError:
        failed = True
    if failed:
        pytest.fail("Name does not exist")
