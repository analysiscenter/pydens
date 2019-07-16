# pylint: disable=redefined-outer-name, missing-docstring
import sys
import pytest

sys.path.append('..')
from batchflow import Config


@pytest.fixture
def config():
    _config = dict(key1='val1', key2=dict())
    _config['key2']['subkey1'] = 'val21'
    return Config(_config)


class TestConfig:
    def test_getitem_key(self, config):
        assert config['key1'] == config.config['key1']

    def test_getitem_missing_key(self, config):
        with pytest.raises(KeyError):
            _ = config['missing key']

    def test_getitem_nested_key(self, config):
        assert config['key2/subkey1'] == config.config['key2']['subkey1']

    def test_get_key(self, config):
        assert config.get('key1') == config.config.get('key1')

    def test_get_nested_key(self, config):
        assert config.get('key2/subkey1') == config.config['key2']['subkey1']

    def test_get_missing_key(self, config):
        assert config.get('missing key') is None

    def test_get_missing_key_with_default(self, config):
        assert config.get('missing key', default=1) == 1

    def test_get_nested_missing_key_with_default(self, config):
        assert config.get('key2/missing key', default=1) == 1

    def test_pop_key(self, config):
        val = config.config.get('key1')
        assert config.pop('key1') == val
        assert 'key1' not in config, 'key should have been deleted'

    def test_pop_nested_key(self, config):
        val = config.config['key2']['subkey1']
        assert config.pop('key2/subkey1') == val
        assert 'subkey1' not in config, 'nested key should have been deleted'
        assert 'key2' in config, 'outer key should remain'

    def test_pop_missing_key(self, config):
        with pytest.raises(KeyError):
            _ = config.pop('missing key')

    def test_pop_missing_key_with_default(self, config):
        assert config.pop('missing key', default=1) == 1

    def test_pop_nested_missing_key_with_default(self, config):
        assert config.pop('key2/missing key', default=1) == 1
