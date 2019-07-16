""" Tests for DeepGalerkin model for solving PDE's. """
import warnings
import pytest

BAD_PREFIXES = ['get_ipython', 'plt', 'axes', 'fig.', 'fig,']

@pytest.mark.slow
def test_run_tutorial():
    """ There are a lot of examples in tutorial, and all of them
    should be working.

    Notes
    -----
    IPython notebook is converted to Python script, so tqdm bars and
    plotting is removed.
    """
    # pylint: disable=exec-used
    tutorials_dir = './../../examples/tutorials/'
    notebook = '09_solving_PDE_with_NN.ipynb'
    file = tutorials_dir + notebook

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nbconvert import PythonExporter
        code, _ = PythonExporter().from_filename(file)

    code_ = []
    for line in code.split('\n'):
        if not line.startswith('#'):
            if not any(bs in line for bs in BAD_PREFIXES):
                code_.append(line.replace('in tqdm_notebook', 'in '))

    code = '\n'.join(code_)
    exec(code, {})
