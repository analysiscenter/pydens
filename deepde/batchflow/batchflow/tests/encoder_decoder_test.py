""" Test for EncoderDecoder model architecture.
First of all, we define possible types of encoders, embeddings and decoders.
Later every combination of encoder, embedding, decoder is combined into one model and we initialize it.
"""
# pylint: disable=import-error, no-name-in-module
# pylint: disable=redefined-outer-name
import pytest

from batchflow.models.tf import EncoderDecoder, VariationalAutoEncoder
from batchflow.models.tf import ResNet, MobileNet, DenseNet



MODELS = [
    EncoderDecoder,
    VariationalAutoEncoder
]


ENCODERS = [
    {'num_stages': 2},
    {'base': ResNet, 'num_blocks': [2]*3, 'filters': [13]*3},
    {'base': DenseNet, 'num_layers': [2]*3, 'growth_rate': 13},
    {'num_stages': 2, 'blocks': {'base': ResNet.block, 'filters':[13]*2}},
]


EMBEDDINGS = [
    {},
    {'base': MobileNet.block, 'width_factor': 2},
]


DECODERS = [
    {},
    {'num_stages': 2, 'factor': 9, 'skip': False, 'upsample': {'layout': 'X'}},
    {'num_stages': 4, 'blocks': {'layout': 'cnacna', 'filters': [23]*4}},
    {'num_stages': 4, 'blocks': {'base': DenseNet.block, 'num_layers': [2]*4, 'growth_rate': 23}},
]


@pytest.fixture()
def base_config():
    """ Fixture to hold default configuration. """
    config = {
        'inputs': {'images': {'shape': (16, 16, 1)},
                   'masks': {'name': 'targets', 'shape': (16, 16, 1)}},
        'initial_block': {'inputs': 'images'},
        'loss': 'mse'
    }
    return config


@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('decoder', DECODERS)
@pytest.mark.parametrize('embedding', EMBEDDINGS)
@pytest.mark.parametrize('encoder', ENCODERS)
def test_first(base_config, model, encoder, embedding, decoder):
    """ Create encoder-decoder architecture from every possible combination
    of encoder, embedding, decoder, listed in global variables defined above.
    """
    base_config.update({'body/encoder': encoder,
                        'body/embedding': embedding,
                        'body/decoder': decoder})
    _ = model(base_config)
