""" Contains basic Batch classes """

import os
import traceback
import threading
import warnings

import dill
try:
    import blosc
except ImportError:
    pass
import numpy as np
try:
    import pandas as pd
except ImportError:
    pass
try:
    import feather
except ImportError:
    pass
try:
    import dask.dataframe as dd
except ImportError:
    pass

from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, inbatch_parallel, any_action_failed
from .components import MetaComponentsTuple


class Batch:
    """ The core Batch class """
    _item_class = None
    components = None

    def __init__(self, index, preloaded=None, *args, **kwargs):
        _ = args
        if  self.components is not None and not isinstance(self.components, tuple):
            raise TypeError("components should be a tuple of strings with components names")
        self.index = index
        self._data_named = None
        self._data = None
        self._preloaded_lock = threading.Lock()
        self._preloaded = preloaded
        self._local = None
        self._pipeline = None
        self.create_attrs(**kwargs)

    def create_attrs(self, **kwargs):
        """ Create attributes from kwargs """
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @property
    def pipeline(self):
        """: Pipeline - a pipeline the batch is being used in """
        if self._local is not None and hasattr(self._local, 'pipeline'):
            return self._local.pipeline
        return self._pipeline

    @pipeline.setter
    def pipeline(self, val):
        """ Store pipeline in a thread-local storage """
        if val is None:
            self._local = None
        else:
            if self._local is None:
                self._local = threading.local()
            self._local.pipeline = val
        self._pipeline = val

    def __copy__(self):
        pipeline = self.pipeline
        self.pipeline = None
        dump_batch = dill.dumps(self)
        self.pipeline = pipeline

        restored_batch = dill.loads(dump_batch)
        restored_batch.pipeline = pipeline
        return restored_batch

    def deepcopy(self):
        """ Return a deep copy of the batch.

        Constructs a new ``Batch`` instance and then recursively copies all
        the objects found in the original batch, except the ``pipeline``,
        which remains unchanged.

        Returns
        -------
        Batch
        """
        return self.copy()

    @classmethod
    def from_data(cls, index, data):
        """ Create batch from a given dataset """
        # this is roughly equivalent to self.data = data
        if index is None:
            index = np.arange(len(data))
        return cls(index, preloaded=data)

    @classmethod
    def from_batch(cls, batch):
        """ Create batch from another batch """
        return cls(batch.index, preloaded=batch._data)  # pylint: disable=protected-access


    @classmethod
    def merge(cls, batches, batch_size=None):
        """ Merge several batches to form a new batch of a given size

        Parameters
        ----------
        batches : tuple of batches

        batch_size : int or None
            if `None`, just merge all batches into one batch (the rest will be `None`),
            if `int`, then make one batch of `batch_size` and a batch with the rest of data.

        Returns
        -------
        batch, rest : tuple of two batches

        Raises
        ------
        ValueError
            If component is `None` in some batches and not `None` in others.
        """
        def _make_index(data):
            return DatasetIndex(np.arange(data.shape[0])) if data is not None and data.shape[0] > 0 else None

        def _make_batch(data):
            index = _make_index(data[0])
            return cls(index, preloaded=tuple(data)) if index is not None else None

        if batch_size is None:
            break_point = len(batches)
            last_batch_len = len(batches[-1])
        else:
            break_point = -1
            last_batch_len = 0
            cur_size = 0
            for i, b in enumerate(batches):
                cur_batch_len = len(b)
                if cur_size + cur_batch_len >= batch_size:
                    break_point = i
                    last_batch_len = batch_size - cur_size
                    break
                else:
                    cur_size += cur_batch_len
                    last_batch_len = cur_batch_len

        components = batches[0].components or (None,)
        new_data = list(None for _ in components)
        rest_data = list(None for _ in components)
        for i, comp in enumerate(components):
            none_components_in_batches = [b.get(component=comp) is None for b in batches]
            if np.all(none_components_in_batches):
                continue
            elif np.any(none_components_in_batches):
                raise ValueError('Component {} is None in some batches'.format(comp))

            if batch_size is None:
                new_comp = [b.get(component=comp) for b in batches[:break_point]]
            else:
                b = batches[break_point]
                last_batch_len_ = b.get_pos(None, comp, b.indices[last_batch_len - 1])
                new_comp = [b.get(component=comp) for b in batches[:break_point]] + \
                           [batches[break_point].get(component=comp)[:last_batch_len_ + 1]]
            new_data[i] = cls.merge_component(comp, new_comp)

            if batch_size is not None:
                rest_comp = [batches[break_point].get(component=comp)[last_batch_len_ + 1:]] + \
                            [b.get(component=comp) for b in batches[break_point + 1:]]
                rest_data[i] = cls.merge_component(comp, rest_comp)

        new_batch = _make_batch(new_data)
        rest_batch = _make_batch(rest_data)

        return new_batch, rest_batch

    @classmethod
    def merge_component(cls, component=None, data=None):
        """ Merge the same component data from several batches """
        _ = component
        if isinstance(data[0], np.ndarray):
            return np.concatenate(data)
        raise TypeError("Unknown data type", type(data[0]))

    def as_dataset(self, dataset):
        """ Makes a new dataset from batch data

        Parameters
        ----------
        dataset
            an instance or a subclass of Dataset

        Returns
        -------
        an instance of a class specified by `dataset` arg, preloaded with this batch data
        """
        if dataset is None:
            raise ValueError('dataset can be an instance of Dataset (sub)class or the class itself, but not None')
        if isinstance(dataset, type):
            dataset_class = dataset
        else:
            dataset_class = dataset.__class__
        return dataset_class(self.index, batch_class=type(self), preloaded=self.data)

    @property
    def indices(self):
        """: numpy array - an array with the indices """
        if isinstance(self.index, DatasetIndex):
            return self.index.indices
        return self.index

    def __len__(self):
        return len(self.index)

    @property
    def size(self):
        """: int - number of items in the batch """
        return len(self)

    @property
    def data(self):
        """: tuple or named components - batch data """
        if self._data is None and self._preloaded is not None:
            # load data the first time it's requested
            with self._preloaded_lock:
                if self._data is None and self._preloaded is not None:
                    self.load(src=self._preloaded)
        res = self._data if self.components is None else self._data_named
        return res if res is not None else self._empty_data

    def make_item_class(self, local=False):
        """ Create a class to handle data components """
        # pylint: disable=protected-access
        if self.components is None:
            type(self)._item_class = None
        elif type(self)._item_class is None or not local:
            comp_class = MetaComponentsTuple(type(self).__name__ + 'Components', components=self.components)
            type(self)._item_class = comp_class
        else:
            comp_class = MetaComponentsTuple(type(self).__name__ + 'Components' + str(id(self)),
                                             components=self.components)
            self._item_class = comp_class

    @action
    def add_components(self, components, init=None):
        """ Add new components

        Parameters
        ----------
        components : str or list
            new component names
        init : array-like
            initial component data

        Raises
        ------
        ValueError
            If a component or an attribute with the given name already exists
        """
        if isinstance(components, str):
            components = (components,)
            init = (init,)
        elif isinstance(components, (tuple, list)):
            components = tuple(components)
            if init is None:
                init = (None,) * len(components)
            else:
                init = tuple(init)

        if any(hasattr(self, c) for c in components):
            raise ValueError("An attribute with the same name exists")

        data = self._data
        if self.components is None:
            self.components = tuple()
            data = tuple()
            warnings.warn("All batch data is erased")

        exists_component = set(components) & set(self.components)
        if exists_component:
            raise ValueError("Component(s) with name(s) '{}' already exists".format("', '".join(exists_component)))

        self.components = self.components + components

        self.make_item_class(local=True)
        if data is not None:
            self._data = data + init

        return self

    def __getattr__(self, name):
        if self.components is not None and name in self.components:   # pylint: disable=unsupported-membership-test
            attr = getattr(self.data, name)
            return attr
        raise AttributeError("%s not found in class %s" % (name, self.__class__.__name__))

    def __setattr__(self, name, value):
        if self.components is not None:
            if name == "_data":
                super().__setattr__(name, value)
                if self._item_class is None:
                    self.make_item_class()
                self._data_named = self._item_class(data=self._data)   # pylint: disable=not-callable
                return
            if name in self.components:    # pylint: disable=unsupported-membership-test
                if self._data_named is None:
                    _ = self.data
                setattr(self._data_named, name, value)
                super().__setattr__('_data', self._data_named.data)
                return
        super().__setattr__(name, value)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_data_named')
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            # this warrants that all hidden objects are reconstructed upon unpickling
            setattr(self, k, v)

    @property
    def _empty_data(self):
        return None if self.components is None else self._item_class()   # pylint: disable=not-callable

    @property
    def array_of_nones(self):
        """1-D ndarray: ``NumPy`` array with ``None`` values."""
        return np.array([None] * len(self.index))

    def get_pos(self, data, component, index):
        """ Return a position in data for a given index

        Parameters
        ----------
        data : some array or tuple of arrays
            if `None`, should return a position in :attr:`self.data <.Batch.data>`

        components : None, int or str
            - None - data has no components (e.g. just an array or pandas.DataFrame)
            - int - a position of a data component, when components names are not defined
                (e.g. data is a tuple)
            - str - a name of a data component

        index : any
            an index id

        Returns
        -------
        int
            a position in a batch data where an item with a given index is stored

        Notes
        -----
        It is used to read / write data from / to a given component::

            batch_data = data.component[pos]
            data.component[pos] = new_data

        if `self.data` holds a numpy array, then get_pos(None, None, index) should
        just return `self.index.get_pos(index)`

        if `self.data.images` contains BATCH_SIZE images as a numpy array,
        then `get_pos(None, 'images', index)` should return `self.index.get_pos(index)`

        if `self.data.labels` is a dict {index: label}, then `get_pos(None, 'labels', index)` should return index.

        if `data` is not `None`, then you need to know in advance how to get a position for a given index.

        For instance, `data` is a large numpy array, and a batch is a subset of this array and
        `batch.index` holds row numbers from a large arrays.
        Thus, `get_pos(data, None, index)` should just return index.

        A more complicated example of data:

        - batch represent small crops of large images
        - `self.data.source` holds a few large images (e.g just 5 items)
        - `self.data.coords` holds coordinates for crops (e.g. 100 items)
        - `self.data.image_no` holds an array of image numbers for each crop (so it also contains 100 items)

        then `get_pos(None, 'source', index)` should return `self.data.image_no[self.index.get_pos(index)]`.
        Whilst, `get_pos(data, 'source', index)` should return `data.image_no[index]`.
        """
        _ = component
        if data is None:
            pos = self.index.get_pos(index)
        else:
            pos = index
        return pos

    def put_into_data(self, data, dst=None):
        """ Load data into :attr:`_data` property """
        if self.components is None:
            _src = data
        else:
            _src = data if isinstance(data, tuple) or data is None else tuple([data])
        _src = self.get_items(self.indices, _src)

        if dst is None:
            self._data = _src
        else:
            components = [dst] if isinstance(dst, str) else dst
            for i, comp in enumerate(components):
                if isinstance(_src, dict):
                    comp_src = _src[comp]
                else:
                    comp_src = _src[i]
                setattr(self, comp, comp_src)

    def get_items(self, index, data=None, components=None):
        """ Return one or several data items from a data source """
        if data is None:
            _data = self.data
        else:
            _data = data
        if components is None:
            components = self.components

        if self._item_class is not None and isinstance(_data, self._item_class):
            pos = [self.get_pos(None, comp, index) for comp in components]   # pylint: disable=not-an-iterable
            res = self._item_class(data=_data, pos=pos)    # pylint: disable=not-callable
        elif isinstance(_data, tuple):
            comps = components if components is not None else range(len(_data))
            res = tuple(data_item[self.get_pos(data, comp, index)] if data_item is not None else None
                        for comp, data_item in zip(comps, _data))
        elif isinstance(_data, dict):
            res = dict(zip(components, (_data[comp][self.get_pos(data, comp, index)] for comp in components)))
        else:
            pos = self.get_pos(data, None, index)
            res = _data[pos]
        return res

    def get(self, item=None, component=None):
        """ Return an item from the batch or the component """
        if item is None:
            if component is None:
                res = self.data
            else:
                res = getattr(self, component)
        else:
            if component is None:
                res = self[item]
            else:
                res = self[item]
                res = getattr(res, component)
        return res

    def __getitem__(self, item):
        return self.get_items(item)

    def __iter__(self):
        for item in self.indices:
            yield self[item]

    @property
    def items(self):
        """: list - batch items """
        return [[self[ix]] for ix in self.indices]

    def run_once(self, *args, **kwargs):
        """ Init function for no parallelism
        Useful for async action-methods (will wait till the method finishes)
        """
        _ = self.data, args, kwargs
        return [[]]

    def get_model_by_name(self, model_name):
        """ Return a model specification given its name """
        return self.pipeline.get_model_by_name(model_name, batch=self)

    def get_errors(self, all_res):
        """ Return a list of errors from a parallel action """
        all_errors = [error for error in all_res if isinstance(error, Exception)]
        return all_errors if len(all_errors) > 0 else None

    @action
    def do_nothing(self, *args, **kwargs):
        """ An empty action (might be convenient in complicated pipelines) """
        _ = args, kwargs
        return self

    @action
    @inbatch_parallel(init='indices', post='_assemble')
    def apply_transform(self, ix, func, *args, src=None, dst=None, p=None, use_self=False, **kwargs):
        """ Apply a function to each item in the batch.

        Parameters
        ----------
        func : callable
            a function to apply to each item from the source

        src : str, sequence, list of str
            the source to get data from, can be:

            - None
            - str - a component name, e.g. 'images' or 'masks'
            - sequence - a numpy-array, list, etc
            - tuple of str - get data from several components
            - list of str, sequences or tuples - apply same transform to each item in list

        dst : str or array
            the destination to put the result in, can be:

            - None - in this case dst is set to be same as src
            - str - a component name, e.g. 'images' or 'masks'
            - tuple of list of str, e.g. ['images', 'masks']

            if src is a list, dst should be either list or None.

        p : float or None
            probability of applying transform to an element in the batch

            if not None, indices of relevant batch elements will be passed ``func``
            as a named arg ``indices``.

        use_self : bool
            whether to pass ``self`` to ``func``

        args, kwargs
            other parameters passed to ``func``

        Notes
        -----
        apply_transform does the following (but in parallel)::

            for item in range(len(batch)):
                self.dst[item] = func(self.src[item], *args, **kwargs)

        If `src` is a list with two or more elements, `dst` should be list or
        tuple of the same lenght.

        Examples
        --------

        ::

            apply_transform(make_masks_fn, src='images', dst='masks')
            apply_transform(apply_mask, src=('images', 'masks'), dst='images', use_self=True)
            apply_transform_all(rotate, src=['images', 'masks'], dst=['images', 'masks'], p=.2)
        """
        dst = src if dst is None else dst

        if not (isinstance(dst, str) or
                (isinstance(dst, (list, tuple)) and np.all([isinstance(component, str) for component in dst]))):
            raise TypeError("dst should be str or tuple or list of str")

        p = 1 if p is None or np.random.binomial(1, p) else 0

        if isinstance(src, list) and len(src) > 1 and isinstance(dst, list) and len(src) == len(dst):
            return tuple([self._apply_transform(ix, func, *args, src=src_component,
                                                dst=dst_component, p=p, use_self=use_self, **kwargs)
                          for src_component, dst_component in zip(src, dst)])
        return self._apply_transform(ix, func, *args, src=src, dst=dst, p=p, use_self=use_self, **kwargs)

    def _apply_transform(self, ix, func, *args, src=None, dst=None, p=None, use_self=False, **kwargs):
        """ Apply a function to each item in the batch.

        Parameters
        ----------
        func : callable
            a function to apply to each item from the source

        src : str, sequence, list of str
            the source to get data from, can be:

            - None
            - str - a component name, e.g. 'images' or 'masks'
            - sequence - a numpy-array, list, etc
            - tuple of str - get data from several components

        dst : str or array
            the destination to put the result in, can be:

            - None
            - str - a component name, e.g. 'images' or 'masks'
            - array-like - a numpy-array, list, etc

        use_self : bool
            whether to pass ``self`` to ``func``

        args, kwargs
            other parameters passed to ``func``
        """
        if src is None:
            _args = args
        else:
            if isinstance(src, str):
                pos = self.get_pos(None, src, ix)
                src_attr = (getattr(self, src)[pos],)
            elif isinstance(src, (tuple, list)) and np.all([isinstance(component, str) for component in src]):
                src_attr = [getattr(self, component)[self.get_pos(None, component, ix)] for component in src]
            else:
                pos = self.get_pos(None, dst, ix)
                src_attr = (src[pos],)
            _args = tuple([*src_attr, *args])

        if p:
            if use_self:
                return func(self, *_args, **kwargs)
            return func(*_args, **kwargs)

        if len(src_attr) == 1:
            return src_attr[0]
        return src_attr

    @action
    def apply_transform_all(self, func, *args, src=None, dst=None, p=None, use_self=False, **kwargs):
        """ Apply a function the whole batch at once

        Parameters
        ----------
        func : callable
            a function to apply to each item from the source

        src : str or array
            the source to get data from, can be:

            - str - a component name, e.g. 'images' or 'masks'
            - array-like - a numpy-array, list, etc

        dst : str or array
            the destination to put the result in, can be:

            - None
            - str - a component name, e.g. 'images' or 'masks'
            - array-like - a numpy-array, list, etc

        p : float or None
            probability of applying transform to an element in the batch

            if not None, indices of relevant batch elements will be passed ``func``
            as a named arg ``indices``.

        use_self : bool
            whether to pass ``self`` to ``func``

        args, kwargs
            other parameters passed to ``func``

        Notes
        -----
        apply_transform_all does the following::

            self.dst = func(self.src, *args, **kwargs)

        When ``p`` is passed, random indices are chosen first and then passed to ``func``::

            self.dst = func(self.src, *args, indices=random_indices, **kwargs)

        Transform functions might be methods as well, when ``use_self=True``::

            self.dst = func(self, self.src, *args, **kwargs)

        Examples
        --------

        ::

            apply_transform_all(make_masks_fn, src='images', dst='masks')
            apply_transform_all(MyBatch.make_masks, src='images', dst='masks', use_self=True)
            apply_transform_all(custom_crop, src='images', dst='augmented_images', p=.2)

        """
        if not isinstance(dst, str) and not isinstance(src, str):
            raise TypeError("At least one of dst and src should be attribute names, not arrays")

        if src is None:
            _args = args
        else:
            if isinstance(src, str):
                src_attr = getattr(self, src)
            else:
                src_attr = src
            _args = tuple([src_attr, *args])

        if p is not None:
            indices = np.where(np.random.binomial(1, p, len(self)))[0]
            kwargs['indices'] = indices
        if use_self:
            _args = (self, *_args)
        tr_res = func(*_args, **kwargs)

        if dst is None:
            pass
        elif isinstance(dst, str):
            setattr(self, dst, tr_res)
        else:
            dst[:] = tr_res
        return self

    def _get_file_name(self, ix, src):
        """ Get full path file name corresponding to the current index.

        Parameters
        ----------
        src : str, FilesIndex or None
            if None, full path to the indexed item will be returned.
            if FilesIndex it must contain the same indices values as in the self.index.
            if str, behavior depends on wheter self.index.dirs is True. If self.index.dirs is True
            then src will be appended to the end of the full paths from self.index. Else if
            self.index.dirs is False then src is considered as a directory name and the basenames
            from self.index will be appended to the end of src.

        Examples
        --------
        Let folder "/some/path/*.dcm" contain files "001.png", "002.png", etc. Then if self.index
        was built as

        >>> index = FilesIndex(path="/some/path/*.png", no_ext=True)

        Then _get_file_name(ix, src="augmented_images/") will return filenames:
        "augmented_images/001.png", "augmented_images/002.png", etc.


        Let folder "/some/path/*" contain folders "001", "002", etc. Then if self.index
        was built as

        >>> index = FilesIndex(path="/some/path/*", dirs=True)

        Then _get_file_name(ix, src="masks.png") will return filenames:
        "/some/path/001/masks.png", "/some/path/002/masks.png", etc.


        If you have two directories "images/*.png", "labels/*png" with identical filenames,
        you can build two instances of FilesIndex and use the first one to biuld your Dataset

        >>> index_images = FilesIndex(path="/images/*.png", no_ext=True)
        >>> index_labels = FilesIndex(path="/labels/*.png", no_ext=True)
        >>> dset = Dataset(index=index_images, batch_class=Batch)

        Then build dataset using the first one
        _get_file_name(ix, src=index_labels) to reach corresponding files in the second path.

        """
        if not isinstance(self.index, FilesIndex):
            raise ValueError("File locations must be specified to dump/load data")

        if isinstance(src, str):
            if self.index.dirs:
                fullpath = self.index.get_fullpath(ix)
                file_name = os.path.join(fullpath, src)

            else:
                file_name = os.path.basename(self.index.get_fullpath(ix))
                file_name = os.path.join(os.path.abspath(src), file_name)

        elif isinstance(src, FilesIndex):
            try:
                file_name = src.get_fullpath(ix)
            except KeyError:
                raise KeyError("File {} is not indexed in the received index".format(ix))

        elif src is None:
            file_name = self.index.get_fullpath(ix)

        else:
            raise ValueError("Src must be either str, FilesIndex or None")

        return file_name

    def _assemble_component(self, result, *args, component, **kwargs):
        """ Assemble one component after parallel execution.

        Parameters
        ----------
        result : sequence, np.ndarray
            Values to put into ``component``
        component : str
            Component to assemble.
        """

        _ = args, kwargs
        try:
            new_items = np.stack(result)
        except ValueError as e:
            message = str(e)
            if "must have the same shape" in message:
                new_items = np.empty(len(result), dtype=object)
                new_items[:] = result
            else:
                raise e
        setattr(self, component, new_items)

    def _assemble(self, all_results, *args, dst=None, **kwargs):
        """ Assembles the batch after a parallel action.

        Parameters
        ----------
        all_results : sequence
            Results after inbatch_parallel.
        dst : str, sequence, np.ndarray
            Components to assemble

        Returns
        -------
        self
        """

        _ = args
        if any_action_failed(all_results):
            all_errors = self.get_errors(all_results)
            print(all_errors)
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch")
        if dst is None:
            dst_default = kwargs.get('dst_default', 'src')
            if dst_default == 'src':
                dst = kwargs.get('src')
            elif dst_default == 'components':
                dst = self.components

        if not isinstance(dst, (list, tuple, np.ndarray)):
            dst = [dst]

        if len(dst) == 1:
            all_results = [all_results]
        else:
            all_results = list(zip(*all_results))

        for component, result in zip(dst, all_results):
            self._assemble_component(result, component=component, **kwargs)
        return self

    @inbatch_parallel('indices', post='_assemble', target='f', dst_default='components')
    def _load_blosc(self, ix, src=None, dst=None):
        """ Load data from a blosc packed file """
        file_name = self._get_file_name(ix, src)
        with open(file_name, 'rb') as f:
            data = dill.loads(blosc.decompress(f.read()))
            components = tuple(dst or self.components)
            try:
                item = tuple(data[i] for i in components)
            except Exception as e:
                raise KeyError('Cannot find components in corresponfig file', e)
        return item

    @inbatch_parallel('indices', target='f')
    def _dump_blosc(self, ix, dst, components=None):
        """ Save blosc packed data to file """
        file_name = self._get_file_name(ix, dst)
        with open(file_name, 'w+b') as f:
            if self.components is None:
                components = (None,)
                item = (self[ix],)
            else:
                components = tuple(components or self.components)
                item = self[ix].as_tuple(components)
            data = dict(zip(components, item))
            f.write(blosc.compress(dill.dumps(data)))

    def _load_table(self, src, fmt, dst=None, post=None, *args, **kwargs):
        """ Load a data frame from table formats: csv, hdf5, feather """
        if fmt == 'csv':
            if 'index_col' in kwargs:
                index_col = kwargs.pop('index_col')
                _data = pd.read_csv(src, *args, **kwargs).set_index(index_col)
            else:
                _data = pd.read_csv(src, *args, **kwargs)
        elif fmt == 'feather':
            _data = feather.read_dataframe(src, *args, **kwargs)
        elif fmt == 'hdf5':
            _data = pd.read_hdf(src, *args, **kwargs)

        # Put into this batch only part of it (defined by index)
        if isinstance(_data, pd.DataFrame):
            _data = _data.loc[self.indices]
        elif isinstance(_data, dd.DataFrame):
            # dask.DataFrame.loc supports advanced indexing only with lists
            _data = _data.loc[list(self.indices)].compute()

        if callable(post):
            _data = post(_data, src=src, fmt=fmt, dst=dst, **kwargs)
        else:
            components = tuple(dst or self.components)
            _new_data = dict()
            for i, comp in enumerate(components):
                _new_data[comp] = _data.iloc[:, i].values
            _data = _new_data

        for comp, values in _data.items():
            setattr(self, comp, values)


    @action(use_lock='__dump_table_lock')
    def _dump_table(self, dst, fmt='feather', components=None, *args, **kwargs):
        """ Save batch data to table formats

        Args:
          dst: str - a path to dump into
          fmt: str - format: feather, hdf5, csv
          components: str or tuple - one or several component names
        """
        filename = dst

        components = tuple(components or self.components)
        data_dict = {}
        for comp in components:
            comp_data = self.get(component=comp)
            if isinstance(comp_data, pd.DataFrame):
                data_dict.update(comp_data.to_dict('series'))
            elif isinstance(comp_data, np.ndarray):
                if comp_data.ndim > 1:
                    columns = [comp + str(i) for i in range(comp_data.shape[1])]
                    comp_dict = zip(columns, (comp_data[:, i] for i in range(comp_data.shape[1])))
                    data_dict.update({comp: comp_dict})
                else:
                    data_dict.update({comp: comp_data})
            else:
                data_dict.update({comp: comp_data})
        _data = pd.DataFrame(data_dict)

        if fmt == 'feather':
            feather.write_dataframe(_data, filename, *args, **kwargs)
        elif fmt == 'hdf5':
            _data.to_hdf(filename, *args, **kwargs)   # pylint:disable=no-member
        elif fmt == 'csv':
            _data.to_csv(filename, *args, **kwargs)   # pylint:disable=no-member
        else:
            raise ValueError('Unknown format %s' % fmt)

        return self

    @action
    def load(self, *args, src=None, fmt=None, dst=None, **kwargs):
        """ Load data from another array or a file.

        Parameters
        ----------
        src :
            a source (e.g. an array or a file name)

        fmt : str
            a source format, one of None, 'blosc', 'csv', 'hdf5', 'feather'

        dst : None or str or tuple of str
            components to load

        **kwargs :
            other parameters to pass to format-specific loaders
        """
        _ = args
        components = [dst] if isinstance(dst, str) else dst
        if components is not None:
            self.add_components(np.setdiff1d(components, self.components).tolist())

        if fmt is None:
            self.put_into_data(src, components)
        elif fmt == 'blosc':
            self._load_blosc(src=src, dst=components, **kwargs)
        elif fmt in ['csv', 'hdf5', 'feather']:
            self._load_table(src=src, fmt=fmt, dst=components, **kwargs)
        else:
            raise ValueError("Unknown format " + fmt)
        return self

    @action
    def dump(self, *args, dst=None, fmt=None, components=None, **kwargs):
        """ Save data to another array or a file.

        Parameters
        ----------
        dst :
            a destination (e.g. an array or a file name)

        fmt : str
            a destination format, one of None, 'blosc', 'csv', 'hdf5', 'feather'

        components : None or str or tuple of str
            components to load

        *args :
            other parameters are passed to format-specific writers

        *kwargs :
            other parameters are passed to format-specific writers
        """
        components = [components] if isinstance(components, str) else components
        if fmt is None:
            if components is not None and len(components) > 1:
                raise ValueError("Only one component can be dumped into a memory array: components =", components)
            components = components[0] if components is not None else None
            dst[self.indices] = self.get(component=components)
        elif fmt == 'blosc':
            self._dump_blosc(dst, components=components)
        elif fmt in ['csv', 'hdf5', 'feather']:
            self._dump_table(dst, fmt, components, *args, **kwargs)
        else:
            raise ValueError("Unknown format " + fmt)
        return self

    @action
    def save(self, *args, **kwargs):
        """ Save batch data to a file (an alias for dump method)"""
        return self.dump(*args, **kwargs)
