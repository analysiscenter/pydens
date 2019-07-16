"""
Each function tests specific Config class method.
"""

import sys
import pytest

sys.path.append('../..')

from batchflow import Config

def test_dict_init():
    """
    Tests Config.__init__() using input of dictionary type.
    For inner structure check Config.flatten() is used.
    """

    #Slashed-structured dictionary initialization
    init_dict = {'a' : 1, 'b/c' : 2, 'b/d' : 3}
    exp_flat = {'a': 1, 'b/c': 2, 'b/d': 3}
    config = Config(init_dict)
    assert config.flatten() == exp_flat

    #Nested-structured dictionary initialization
    init_dict = {'a' : {}, 'b' : {'c' : 2, 'd' : 3}}
    exp_flat = {'a': {}, 'b/c': 2, 'b/d': 3}
    config = Config(init_dict)
    assert config.flatten() == exp_flat

    #Mixed-structured dictionary initialization
    init_dict = {'a' : None, 'b/c' : 2, 'b' : {'d' : 3}}
    exp_flat = {'a': None, 'b/c': 2, 'b/d': 3}
    config = Config(init_dict)
    assert config.flatten() == exp_flat

    #Config-structured dictionary initialization
    init_dict = {'a' : Config({'b' : 2})}
    exp_flat = {'a/b': 2}
    config = Config(init_dict)
    assert config.flatten() == exp_flat

def test_dict_init_bad():
    """
    Tests Config.__init__() using BAD input of dictionary type.
    """

    #Int-keyed dictionary initialization
    init_dict = {0 : 1}
    with pytest.raises(TypeError):
        Config(init_dict)

    #Bool-keyed dictionary initialization
    init_dict = {False : True}
    with pytest.raises(TypeError):
        Config(init_dict)

def test_list_init():
    """
    Tests Config.__init__() using input of list type.
    For inner structure check Config.flatten() is used.
    """

    #Slashed-structured list initialization
    init_list = [('a', 1), ('b/c', 2), ('b/d', 3)]
    exp_flat = {'a': 1, 'b/c': 2, 'b/d': 3}
    config = Config(init_list)
    assert config.flatten() == exp_flat

    #Nested-structured list initialization
    init_list = [('a', {}), ('b', {'c' : 2, 'd' : 3})]
    exp_flat = {'a': {}, 'b/c': 2, 'b/d': 3}
    config = Config(init_list)
    assert config.flatten() == exp_flat

    #Mixed-structured list initialization
    init_list = [('a', None), ('b/c', 2), ('b', {'d' : 3})]
    exp_flat = {'a': None, 'b/c': 2, 'b/d': 3}
    config = Config(init_list)
    assert config.flatten() == exp_flat

    #Config-structured list initialization
    init_list = [('a', Config({'b' : 2}))]
    exp_flat = {'a/b': 2}
    config = Config(init_list)
    assert config.flatten() == exp_flat

def test_list_init_bad():
    """
    Tests Config.__init__() using BAD input of list type.
    """

    #Int-keyed list initialization
    init_list = [(0, 1)]
    with pytest.raises(TypeError):
        Config(init_list)

    #Bool-keyed list initialization
    init_list = [(False, True)]
    with pytest.raises(TypeError):
        Config(init_list)

    #Bad-shaped list initialization
    init_list = [('a', 0, 1)]
    with pytest.raises(ValueError):
        Config(init_list)

def test_config_init():
    """
    Tests Config.__init__() using input of Config type.
    For inner structure check Config.flatten() is used.
    """

    #Basically, there nothing to test here,
    #but since Config can be initialized with its own instance...
    init_config = Config({'a': 0})
    exp_flat = {'a' : 0}
    config = Config(init_config)
    assert config.flatten() == exp_flat

def test_pop():
    """
    Tests Config.pop(), comparing the return value with expected one.
    For inner structure check Config.flatten() is used.
    """

    #Pop scalar value by slashed-structured key
    config = Config({'a' : 1, 'b/c' : 2, 'b/d' : 3})
    pop_key = 'b/c'
    exp_ret = 2
    exp_flat = {'a' : 1, 'b/d' : 3}
    assert config.pop(pop_key) == exp_ret
    assert config.flatten() == exp_flat

    #Pop dict value by simple key
    config = Config({'a' : 1, 'b/c' : 2, 'b/d' : 3})
    pop_key = 'b'
    exp_ret = {'c' : 2, 'd' : 3}
    exp_flat = {'a' : 1}
    assert config.pop(pop_key) == exp_ret
    assert config.flatten() == exp_flat

def test_get():
    """
    Tests Config.get(), comparing the return value with expected one.
    For inner structure check Config.flatten() is used.
    """
    #Get scalar value by slashed-structured key
    config = Config({'a' : {'b' : 1}})
    get_key = 'a/b'
    exp_ret = 1
    exp_flat = {'a/b' : 1}
    assert config.get(get_key) == exp_ret
    assert config.flatten() == exp_flat

    #Get dict value by simple key
    config = Config({'a' : {'b' : 1}})
    get_key = 'a'
    exp_ret = {'b' : 1}
    exp_flat = {'a/b' : 1}
    assert config.get(get_key) == exp_ret
    assert config.flatten() == exp_flat

def test_put():
    """
    Tests Config.put(), placing value by key in Config instance.
    For inner structure check Config.flatten() is used.
    """

    #Put scalar value by simple key
    config = Config({'a' : 1})
    put_key = 'b'
    put_val = 2
    exp_flat = {'a' : 1, 'b' : 2}
    config.put(put_key, put_val)
    assert config.flatten() == exp_flat

    #Put scalar value by slashed-structured key
    config = Config({'a/b' : 1})
    put_key = 'a/c'
    put_val = 2
    exp_flat = {'a/b' : 1, 'a/c' : 2}
    config.put(put_key, put_val)
    assert config.flatten() == exp_flat

    #Put dict value by simple key
    config = Config({'a/b' : 1})
    put_key = 'a'
    put_val = {'c' : 2}
    exp_flat = {'a/b' : 1, 'a/c' : 2}
    config.put(put_key, put_val)
    assert config.flatten() == exp_flat

def test_flatten():
    """
    Tests Config.flatten()
    """

    #Flatten none config
    config = Config(None)
    exp_flat = {}
    assert config.flatten() == exp_flat

    #Flatten empty config
    config = Config({})
    exp_flat = {}
    assert config.flatten() == exp_flat

    #Flatten simple config
    config = Config({'a' : 1})
    exp_flat = {'a' : 1}
    assert config.flatten() == exp_flat

    #Flatten nested config
    config = Config({'a' : {'b' : {}, 'c' : {'d' : None}}})
    exp_flat = {'a/b' : {}, 'a/c/d' : None}
    assert config.flatten() == exp_flat

def test_add():
    """
    Tests Config.add(), adding up two Config instances.
    For result inner structure check Config.flatten() is used.
    """

    #Simple summands with non-empty intersection
    augend = Config({'a' : 1, 'b' : 2})
    addend = Config({'b' : 3, 'c' : 4})
    exp_flat = {'a' : 1, 'b' : 3, 'c' : 4}
    result = augend + addend
    assert result.flatten() == exp_flat

    #Nested summands with non-empty intersection
    augend = Config({'a/b' : 1, 'a/c' : 2})
    addend = Config({'a/c/d' : 3, 'e/f' : 4})
    exp_flat = {'a/b' : 1, 'a/c/d' : 3, 'e/f' : 4}
    result = augend + addend
    assert result.flatten() == exp_flat

    #Nested summands with non-standard values such as None and empty dict
    augend = Config({'a/b' : 1, 'b/d' : {}})
    addend = Config({'a' : {}, 'b/d' : None})
    exp_flat = {'a/b': 1, 'b/d': None}
    result = augend + addend
    assert result.flatten() == exp_flat

def test_items():
    """
    Tests Config.items()
    For dict_items conversion cast to list is used.
    """

    #Simple
    config = Config({'a' : 1})
    exp_full = [('a', 1)]
    exp_flat = [('a', 1)]
    assert list(config.items(flatten=False)) == exp_full
    assert list(config.items(flatten=True)) == exp_flat

    #Nested
    config = Config({'a' : {'b' : 1, 'c' : 2}})
    exp_full = [('a', {'b' : 1, 'c' : 2})]
    exp_flat = [('a/b', 1), ('a/c', 2)]
    assert list(config.items(flatten=False)).sort() == exp_full.sort()
    assert list(config.items(flatten=True)).sort() == exp_flat.sort()

    #Deeply nested
    config = Config({'a' : {'b' : 1, 'c' : {'d' : 2}}})
    exp_full = [('a', {'b' : 1, 'c' : {'d' : 2}})]
    exp_flat = [('a/b', 1), ('a/c/d', 2)]
    assert list(config.items(flatten=False)).sort() == exp_full.sort()
    assert list(config.items(flatten=True)).sort() == exp_flat.sort()

def test_keys():
    """
    Tests Config.keys()
    For dict_keys conversion cast to list is used.
    """

    #Simple
    config = Config({'a' : 1})
    exp_full = ['a']
    exp_flat = ['a']
    assert list(config.keys(flatten=False)).sort() == exp_full.sort()
    assert list(config.keys(flatten=True)).sort() == exp_flat.sort()

    #Nested
    config = Config({'a' : {'b' : 1, 'c' : 2}})
    exp_full = ['a']
    exp_flat = ['a/b', 'a/c']
    assert list(config.keys(flatten=False)).sort() == exp_full.sort()
    assert list(config.keys(flatten=True)).sort() == exp_flat.sort()

    #Deeply nested
    config = Config({'a' : {'b' : 1, 'c' : {'d' : 2}}})
    exp_full = ['a']
    exp_flat = ['a/b', 'a/c/d']
    assert list(config.keys(flatten=False)).sort() == exp_full.sort()
    assert list(config.keys(flatten=True)).sort() == exp_flat.sort()

def test_values():
    """
    Tests Config.values()
    For dict_values conversion cast to list is used.
    """

    #Simple
    config = Config({'a' : 1})
    exp_full = [1]
    exp_flat = [1]
    assert list(config.values(flatten=False)).sort() == exp_full.sort()
    assert list(config.values(flatten=True)).sort() == exp_flat.sort()

    #Nested
    config = Config({'a' : {'b' : 1, 'c' : 2}})
    exp_full = [{'b' : 1, 'c' : 2}]
    exp_flat = [1, 2]
    assert list(config.values(flatten=False)).sort() == exp_full.sort()
    assert list(config.values(flatten=True)).sort() == exp_flat.sort()

    #Deeply nested
    config = Config({'a' : {'b' : 1, 'c' : {'d' : 2}}})
    exp_full = [{'b' : 1, 'c' : {'d' : 2}}]
    exp_flat = [1, 2]
    assert list(config.values(flatten=False)).sort() == exp_full.sort()
    assert list(config.values(flatten=True)).sort() == exp_flat.sort()

def test_update():
    """
    Tests Config.update()
    For inner structure check Config.flatten() is used.
    """

    #Value replacement by slashed-structured key
    config_old = Config({'a/b' : 1, 'a/c' : 2})
    config_new = Config({'a/c' : 3, 'a/d' : 4})
    exp_flat = {'a/b' : 1, 'a/c' : 3, 'a/d' : 4}
    config_old.update(config_new)
    assert config_old.flatten() == exp_flat

    #Value insertion by slashed-structured key
    config_old = Config({'a/b' : 1})
    config_new = Config({'a/c/d' : 2})
    exp_flat = {'a/b' : 1, 'a/c/d' : 2}
    config_old.update(config_new)
    assert config_old.flatten() == exp_flat

    #Update with Config instance including None and empty dict values
    config_old = Config({'a' : {}, 'b' : None})
    config_new = Config({'a' : None, 'b' : {}})
    config_old.update(config_new)
    exp_flat = {'a' : None, 'b' : {}}
    assert config_old.flatten() == exp_flat
