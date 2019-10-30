""" Tests for PyDEns-module. """
import os
import warnings
from glob import glob

import pytest

BAD_PREFIXES = ['get_ipython', 'plt', 'axes', 'fig.', 'fig,',
                'ipyw', 'interact(']

NOTEBOOKS = glob(os.path.join(os.path.dirname(__file__), '../../tutorials/*.ipynb'))

@pytest.mark.slow
@pytest.mark.parametrize('path', NOTEBOOKS)
def test_run_tutorial(path):
    """ There are a lot of examples in tutorial, and all of them
    should be working.

    Notes
    -----
    IPython notebook is converted to Python script, so tqdm bars and
    plotting is removed.
    """
    # pylint: disable=exec-used
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nbconvert import PythonExporter        # pylint: disable=import-outside-toplevel
        code, _ = PythonExporter().from_filename(path)

        code_ = []
        for line in code.split('\n'):
            if not line.startswith('#'):
                if not any(bs in line for bs in BAD_PREFIXES):
                    line = line.replace("bar='notebook'", "bar=False")
                    line = line.replace("in tqdm_notebook", "in ")
                    code_.append(line)

        code = '\n'.join(code_)
        exec(code, {})
