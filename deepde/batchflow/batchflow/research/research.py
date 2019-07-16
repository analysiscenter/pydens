""" Class Research and auxiliary classes for multiple experiments. """

import os
import glob
from copy import copy, deepcopy
from collections import OrderedDict
from math import ceil
import json
import dill
import pandas as pd

from .. import Config, Pipeline
from .distributor import Distributor
from .workers import PipelineWorker
from .grid import Grid, Option
from .job import Job
from .utils import get_metrics

class Research:
    """ Class Research for multiple parallel experiments with pipelines. """
    def __init__(self):
        self.executables = OrderedDict()
        self.loaded = False
        self.branches = 1
        self.trails = 3
        self.workers = 1
        self.bar = False
        self.n_reps = 1
        self.name = 'research'
        self.worker_class = PipelineWorker
        self.gpu = None
        self.n_jobs = None
        self.jobs = None
        self.grid_config = None
        self.n_iters = None
        self.timeout = 5
        self.n_splits = None
        self.shuffle = None

    def add_pipeline(self, root, branch=None, dataset=None, part=None, variables=None,
                     name=None, execute='%1', dump=-1, run=False, logging=False, **kwargs):
        """ Add new pipeline to research. Pipeline can be divided into root and branch. In that case root pipeline
        will prepare batch that can be used by different branches with different configs.

        Parameters
        ----------
        root : Pipeline
            a pipeline to execute when the research is run. It must contain `run` action with `lazy=True`.
            Only if `branch` is None, `root` may contain parameters that can be defined by grid.
        branch : Pipeline or None
            a parallelized pipeline to execute within the research.
            Several copies of branch pipeline will be executed in parallel per each batch
            received from the root pipeline.
            May contain parameters that can be defined by grid.
        dataset : Dataset or None
            dataset that will be used with pipelines (see also `part`). If None, root or branch
            must contain datatset.
        part : str or None
            part of dataset to use (for example, `train`)
        variables : str, list of str or None
            names of pipeline variables to save after each iteration into results. All of them must be
            defined in `root` if `branch` is None or be defined in `branch` if `branch` is not None.
            If None, pipeline will be executed without any dumping.
        name : str
            pipeline name. If None, pipeline will have name `pipeline_{index}`
        execute : int, str or list of int or str
            If -1, pipeline will be executed just at last iteration (if `iteration + 1 == n_iters`
            or `StopIteration` was raised)

            If positive int, pipeline will be excuted for that iteration

            If str, must be `'%{step}'` where step is int and function will be executed each `step` iterations.

            If list, must be list of int or str descibed above
        dump : int, str or list of int or str
            iteration when results will be dumped and cleared. Similar to `execute`
        run : bool
            if False then `.next_batch()` will be applied to pipeline, else `.run()` and then `.reset_iter()`.
        kwargs :
            parameters in pipeline config that depends on the names of the other pipeline.

            For example,
            if test pipeline imports model from the other pipeline with name `'train'` in Research,
            corresponding parameter in `import_model` must be `C('import_from')` and `add_pipeline`
            must be called with parameter `import_from='train'`.
        logging : bool
            include execution information to log file or not


        **How to define changing parameters**

        All parameters in `root` or `branch` that are defined in grid should be defined
        as `C('parameter_name')`. Corresponding parameter in grid must have the same `'parameter_name'`.
        """
        name = name or 'pipeline_' + str(len(self.executables) + 1)

        if name in self.executables:
            raise ValueError('Executable unit with name {} was alredy existed'.format(name))

        unit = Executable()
        unit.add_pipeline(root, name, branch, dataset, part, variables, execute, dump, run, logging, **kwargs)
        self.executables[name] = unit
        return self

    def get_metrics(self, pipeline, metrics_var, metrics_name,
                    returns=None, execute='%1', dump=-1, logging=False):
        """ Evaluate metrics.

        Parameters
        ----------
        pipeline : str
            pipeline name
        metrics_var : str
            pipeline variable which accumulate metrics
        metrics_name : str or list of str
            metrics to evaluate
        returns : str, list of str or None
            names to save metrics into results
            if None, `function` will be executed without any saving results and dumping
        execute : int, str or list of int or str
            If -1, metrics will be evaluated just at last iteration (if `iteration + 1 == n_iters`
            or `StopIteration` was raised)

            If positive int, metrics will be evaluated for that iteration

            If str, must be `'%{step}'` where step is int and metrics will be evaluated each `step` iterations.

            If list, must be list of int or str descibed above
        dump : int, str or list of int or str
            iteration when results will be dumped and cleared. Similar to execute
        logging : bool
            include execution information to log file or not
        """
        name = pipeline + '_metrics'
        self.add_function(get_metrics, returns, name, execute, dump,
                          False, logging, pipeline=pipeline,
                          metrics_var=metrics_var, metrics_name=metrics_name)
        return self

    def add_function(self, function, returns=None, name=None, execute='%1', dump=-1,
                     on_root=False, logging=False, *args, **kwargs):
        """ Add function to research.

        Parameters
        ----------
        function : callable
            callable object with following parameters:
                experiment : `OrderedDict` of Executable objects
                    all pipelines and functions that were added to Research
                iteration : int
                    iteration when function is called
                **args, **kwargs
        returns : str, list of str or None
            names for function returns to save into results
            if None, `function` will be executed without any saving results and dumping
        name : str (default None)
            function name. If None, a function will have name `func_{index}`
        execute : int, str or list of int or str
            If -1, function will be executed just at last iteration (if `iteration + 1 == n_iters`
            or `StopIteration` was raised)

            If positive int, function will be excuted for that iteration

            If str, must be `'%{step}'` where step is int and pipeline will be executed each `step` iterations.

            If list, must be list of int or str descibed above
        dump : int, str or list of int or str
            iteration when results will be dumped and cleared. Similar to execute
        on_root : bool
            if False, function will be called with parameters `(iteration, experiment, *args, **kwargs)`,
            else with `(iteration, experiments, *args, **kwargs)` where `experiments` is a list of single experiments
        logging : bool
            include execution information to log file or not
        args : list
            args for the function
        kwargs : dict
            kwargs for the function


        **How to use experiment**

        Each pipeline and function added to Research is saved as an :class:`~.Executable`.

        Experiment is an `OrderedDict` of all pipelines and functions that were added to Research
        and are running in current Job. Key is a name of `Executable`, value is `Executable`.
        """

        name = name or 'func_' + str(len(self.executables) + 1)

        if name in self.executables:
            raise ValueError('Executable unit with name {} was alredy existed'.format(name))

        if on_root and returns is not None:
            raise ValueError("If function on root, then it mustn't have returns")

        unit = Executable()
        unit.add_function(function, name, execute, dump,
                          returns, on_root, logging, *args, **kwargs)
        self.executables[name] = unit

        return self

    def add_grid(self, grid_config):
        """ Add grid of pipeline parameters. Configs from that grid will be generated
        and then substitute into pipelines.

        Parameters
        ----------
        grid_config : dict, Grid or Option
            if dict it should have items parameter_name: list of values.
        """
        self.grid_config = Grid(grid_config)
        return self

    def load_results(self, *args, **kwargs):
        """ Load results of research as pandas.DataFrame or dict (see Results.load). """
        return Results(research=self).load(*args, **kwargs)

    def _create_jobs(self, n_reps, n_iters, folds, branches, name):
        """ Create generator of jobs. If `branches=1` or `len(branches)=1` then each job is one repetition
        for each config from grid_config. Else each job contains several pairs `(repetition, config)`.

        Parameters
        ----------
        n_reps : int

        n_iters : int

        branches : int or list of Configs
            if int, branches is a number of branches for one root
            if list then each element is additional Configs for pipelines
        name : str
            name of research.
        """
        if isinstance(branches, int):
            n_models = branches
        elif branches is None:
            n_models = 1
        else:
            n_models = len(branches)

        folds = range(folds) if isinstance(folds, int) else [None]

        # Create all combinations of possible paramaters, cv partitions and indices of repetitions
        configs_with_repetitions = [(idx, configs, cv_split)
                                    for idx in range(n_reps)
                                    for configs in self.grid_config.gen_configs()
                                    for cv_split in folds]

        # Split all combinations into chunks that will use the same roots
        configs_chunks = self._chunks(configs_with_repetitions, n_models)

        jobs = (Job(self.executables, n_iters,
                    list(zip(*chunk))[0], list(zip(*chunk))[1], list(zip(*chunk))[2],
                    branches, name)
                for chunk in configs_chunks
               )

        n_jobs = ceil(len(configs_with_repetitions) / n_models)

        return jobs, n_jobs

    def _chunks(self, array, size):
        """ Divide array into chunks of the fixed size.

        Parameters
        ----------
        array : list or np.ndarray

        size : int
            chunk size
        """
        for i in range(0, len(array), size):
            yield array[i:i + size]

    def _cv_split(self, n_splits, shuffle):
        has_dataset = False
        for unit in self.executables:
            if getattr(self.executables[unit], 'dataset', None):
                has_dataset = True
                self.executables[unit].dataset.cv_split(n_splits=n_splits, shuffle=shuffle)
        if not has_dataset:
            raise ValueError('At least one pipeline must have dataset to perform cross-validation')

    def run(self, n_reps=1, n_iters=None, workers=1, branches=1, n_splits=None, shuffle=False, name=None,
            bar=False, gpu=None, worker_class=None, timeout=5, trails=2):

        """ Run research.

        Parameters
        ----------
        n_reps : int
            number of repetitions with each combination of parameters from `grid_config`
        n_iters: int or None
            number of iterations for each configurations. If None, wait StopIteration exception for at least
            one pipeline.
        workers : int or list of dicts (Configs)
            If int - number of workers to run pipelines or workers that will run them, `PipelineWorker` will be used.

            If list of dicts (Configs) - list of additional configs which will be appended to configs from tasks.

            Each element corresponds to one worker.
        branches: int or list of dicts (Configs)
            Number of different branches with different configs with the same root. Each branch will use the same batch
            from `root`. Pipelines will be executed in different threads.

            If int - number of pipelines with different configs that will use the same prepared batch
            from `root`.

            If list of dicts (Configs) - list of dicts with additional configs to each pipeline.
        n_splits : int or None
            number of folds for cross-validation.
        shuffle : bool
            cross-validation parameter
        name : str or None
            name folder to save research. By default is 'research'.
        bar : bool or callable
            Whether to show a progress bar.
            If callable, it must have the same signature as `tqdm`.
        gpu : str, list or None
            all gpu devices available for the research.

            Must be of length 1 or be divisible
            by the number of workers.

            If is divisible by the number of workers then
            `length / n_workers` must be 1 or be divisible by the number of branches. If you want to use different
            devices in branches, use expression `C('device')`.

            For example, for :class:`~.TFModel` add `device=C('device')` to model config.
            if None, default gpu configuration will be used
        timeout : int
            each job will be killed if it doesn't answer more then that time in minutes
        trails : int
            trails to execute job

        **How does it work**

        At each iteration all pipelines and functions will be executed in the order in which were added.
        """
        if not self.loaded:
            self.n_reps = n_reps
            self.n_iters = n_iters
            self.workers = workers
            self.branches = branches
            self.bar = bar
            self.gpu = self._get_gpu_list(gpu)
            self.worker_class = worker_class or PipelineWorker
            self.timeout = timeout
            self.trails = trails
            self.name = name or self.name

            self.n_splits = n_splits
            self.shuffle = shuffle

        n_workers = self.workers if isinstance(self.workers, int) else len(self.workers)
        n_branches = self.branches if isinstance(self.branches, int) else len(self.branches)

        if n_splits is not None:
            self._cv_split(n_splits, shuffle)

        if self.grid_config is None:
            self.grid_config = Grid(Option('_dummy', [None]))

        if len(self.gpu) > 1 and len(self.gpu) % n_workers != 0:
            raise ValueError("Number of gpus must be 1 or be divisible \
                             by the number of workers but {} was given".format(len(self.gpu)))

        if len(self.gpu) > 1 and len(self.gpu) // n_workers > 1 and (len(self.gpu) // n_workers) % n_branches != 0:
            raise ValueError("Number of gpus / n_workers must be 1 \
                             or be divisible by the number of branches but {} was given".format(len(self.gpu)))

        self._folder_exists(self.name)

        print("Research {} is starting...".format(self.name))

        self.save()

        self.jobs, self.n_jobs = self._create_jobs(self.n_reps, self.n_iters, self.n_splits, self.branches, self.name)

        distr = Distributor(self.workers, self.gpu, self.worker_class, self.timeout, self.trails)
        distr.run(self.jobs, dirname=self.name, n_jobs=self.n_jobs,
                  n_iters=self.n_iters, bar=self.bar)
        return self

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def _get_gpu_list(self, gpu):
        if gpu is None:
            gpu = []
        elif isinstance(gpu, str):
            gpu = [int(item) for item in gpu.split(',')]
        else:
            gpu = gpu
        return gpu

    def _folder_exists(self, name):
        if not os.path.exists(name):
            os.makedirs(name)
        else:
            raise ValueError(
                "Research with name '{}' already exists".format(name)
            )

    def save(self):
        """ Save description of the research to folder name/description. """
        dirname = os.path.join(self.name, 'description')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(os.path.join(self.name, 'description', 'research.dill'), 'wb') as file:
            dill.dump(self, file)
        with open(os.path.join(self.name, 'description', 'research.json'), 'w') as file:
            file.write(json.dumps(self._json(), default=self._set_default_json))
        with open(os.path.join(self.name, 'description', 'alias.json'), 'w') as file:
            file.write(json.dumps(self.grid_config.description(), default=self._set_default_json))

    def _set_default_json(self, obj):
        try:
            x = json.dumps(obj)
        except TypeError:
            x = str(obj)
        return x

    def _json(self):
        description = copy(self.__dict__)
        description['grid_config'] = self.grid_config.value()
        return description

    @classmethod
    def load(cls, name):
        """ Load description of the research from name/description. """
        with open(os.path.join(name, 'description', 'research.dill'), 'rb') as file:
            res = dill.load(file)
            res.loaded = True
            return res


class Executable:
    """ Function or pipeline

    **Attributes**

    function : callable
        is None if `Executable` is a pipeline
    pipeline : Pipeline
        is None if `Executable` is a function
    root_pipeline : Pipeline
        is None if `Executable` is a function or pipeline is not divided into root and branch
    dataset : Dataset or None
        dataset for pipelines
    part : str or None
        part of dataset to use
    cv_split : int or None
        partition of dataset
    result : dict
        current results of the `Executable`. Keys are names of variables (for pipeline)
        or returns (for function) values are lists of variable values
    path : str
        path to the folder where results will be dumped
    exec : int, list of ints or None
    dump : int, list of ints or None
    to_run : bool
    variables : list
        variables (for pipeline) or returns (for function)
    on_root : bool

    args : list

    kwargs : dict()
    """
    def __init__(self):
        self.function = None
        self.pipeline = None
        self.name = None
        self.result = None
        self.execute = None
        self.dump = None
        self.to_run = None
        self.variables = []
        self.root_pipeline = None
        self.on_root = None
        self.args = []
        self.kwargs = dict()
        self.path = None
        self.config = None
        self.logging = None
        self.additional_config = None
        self.action = None
        self.cv_split = None
        self.dataset = None
        self.part = None

    def add_function(self, function, name, execute='%1', dump=-1, returns=None,
                     on_root=False, logging=False, *args, **kwargs):
        """ Add function as an Executable Unit. """
        returns = returns or []

        if not isinstance(returns, list):
            returns = [returns]

        self.name = name
        self.function = function
        self.execute = execute
        self.dump = dump
        self.variables = returns
        self.args = args
        self.kwargs = kwargs
        self.on_root = on_root
        self.logging = logging

        self.action = {
            'type': 'function',
            'name': name,
            'on_root': on_root
        }

        self._clear_result()
        self._process_iterations()

    def add_pipeline(self, root, name, branch=None, dataset=None, part=None, variables=None,
                     execute='%1', dump=-1, run=False, logging=False, **kwargs):
        """ Add pipeline as an Executable Unit """
        variables = variables or []

        if not isinstance(variables, list):
            variables = [variables]

        if branch is None:
            pipeline = root
            root_pipeline = None
        else:
            pipeline = branch
            root_pipeline = root

        self.name = name
        self.pipeline = pipeline
        self.root_pipeline = root_pipeline
        self.dataset = dataset
        self.part = part
        self.variables = variables
        self.execute = execute
        self.dump = dump
        self.to_run = run
        self.kwargs = kwargs
        self.logging = logging

        self.action = {
            'type': 'pipeline',
            'name': name,
            'root': root is not None,
            'run': run
        }

        self.config = None

        self.additional_config = None

        self._clear_result()
        self._process_iterations()

    def _process_iterations(self):
        if not isinstance(self.execute, list):
            self.execute = [self.execute]
        if not isinstance(self.dump, list):
            self.dump = [self.dump]

    def get_copy(self):
        """ Create copy of unit """
        new_unit = copy(self)
        if self.pipeline is not None:
            new_unit.pipeline += Pipeline()
        new_unit.result = deepcopy(new_unit.result)
        new_unit.variables = copy(new_unit.variables)
        return new_unit

    def set_dataset(self):
        """ Add dataset to root if root exists or create root pipeline on the base of dataset. """
        if self.dataset is not None:
            fold = getattr(self.dataset, 'cv'+str(self.cv_split)) if self.cv_split is not None else self.dataset
            dataset = getattr(fold, self.part) if self.part else fold
            if self.root_pipeline is not None:
                self.root_pipeline = self.root_pipeline << dataset
            else:
                self.pipeline = self.pipeline << dataset

    def reset_iter(self):
        """ Reset iterators in pipelines """
        if self.pipeline is not None:
            self.pipeline.reset_iter()
        if self.root_pipeline is not None:
            self.root_pipeline.reset_iter()

    def _clear_result(self):
        self.result = {var: [] for var in self.variables}
        self.result['iteration'] = []

    def set_config(self, config, worker_config, branch_config, import_config):
        """ Set new config for pipeline """
        self.config = config
        self.additional_config = Config(worker_config) + Config(branch_config) + Config(import_config)

        if self.pipeline is not None:
            self.pipeline.set_config(config.config() + self.additional_config)

    def next_batch(self):
        """ Next batch from pipeline """
        if self.pipeline is not None:
            batch = self.pipeline.next_batch()
        else:
            raise TypeError("Executable should be pipeline, not a function")
        return batch

    def run(self):
        """ Run pipeline """
        if self.pipeline is not None:
            self.pipeline.reset_iter()
            self.pipeline.run()
        else:
            raise TypeError("Executable should be pipeline, not a function")

    def reset_root_iter(self):
        """ Reset pipeline iterator """
        if self.root_pipeline is not None:
            self.root_pipeline.reset_iter()
        else:
            raise TypeError("Executable must have root")

    def next_batch_root(self):
        """ Next batch from root pipeline """
        if self.root_pipeline is not None:
            batch = self.root_pipeline.next_batch()
        else:
            raise TypeError("Executable should have root pipeline")
        return batch

    def execute_for(self, batch, iteration):
        """ Execute pipeline for batch from root """
        _ = iteration
        if self.pipeline is not None:
            batch = self.pipeline.execute_for(batch)
        else:
            raise TypeError("Executable should be pipeline, not a function")
        return batch

    def _call_pipeline(self, iteration, *args, **kwargs):
        _ = args, kwargs
        if self.to_run:
            self.run()
        else:
            self.next_batch()
        self.put_result(iteration)

    def _call_function(self, iteration, *args, **kwargs):
        result = self.function(iteration, *args, **kwargs)
        self.put_result(iteration, result)
        return result

    def __call__(self, iteration, *args, **kwargs):
        if self.pipeline is not None:
            self._call_pipeline(iteration, *args, **kwargs)
        else:
            self._call_function(iteration, *args, **kwargs)

    def put_result(self, iteration, result=None):
        """ Put result from pipeline to self.result """
        if len(self.variables) > 0:
            if self.pipeline is not None:
                for variable in self.variables:
                    self.result[variable].append(self.pipeline.get_variable(variable))
            else:
                if len(self.variables) == 1:
                    result = [result]
                for variable, value in zip(self.variables, result):
                    self.result[variable].append(value)
            self.result['iteration'].append(iteration)

    def dump_result(self, iteration, filename):
        """ Dump pipeline results """
        if len(self.variables) > 0:
            path = os.path.join(self.path, filename + '_' + str(iteration))
            with open(path, 'wb') as file:
                dill.dump(self.result, file)
        self._clear_result()

    def create_folder(self, name):
        """ Create folder if it doesn't exist """
        cv_folder = 'cv_' + str(self.cv_split) if self.cv_split is not None else ''
        self.path = os.path.join(name, 'results', self.config.alias(as_string=True),
                                 str(self.repetition), cv_folder)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def action_iteration(self, iteration, n_iters=None, action='execute'):
        """ Returns does Unit should be executed at that iteration """
        rule = self.execute if action == 'execute' else self.dump
        list_rule = [item for item in rule if isinstance(item, int)]
        step_rule = [int(item[1:]) for item in rule if isinstance(item, str)]

        in_list = iteration in list_rule
        in_step = sum([(iteration+1) % item == 0 for item in step_rule])

        if n_iters is None:
            action_list = in_list or in_step
        else:
            in_final = -1 in list_rule and iteration+1 == n_iters
            action_list = in_list or in_step or in_final
        return action_list

class Results():
    """ Class for dealing with results of research

    Parameters
    ----------
    path : str
        path to root folder of research
    research : Research
        instance of Research
    """
    def __init__(self, path=None, research=None):
        if path is None and research is None:
            raise ValueError('At least one of parameters path and research must be not None')
        if path is None:
            self.research = research
            self.path = research.name
        else:
            self.research = Research().load(path)
            self.path = path

    def _get_list(self, value):
        if not isinstance(value, list):
            value = [value]
        return value

    def _sort_files(self, files, iterations):
        files = {file: int(file.split('_')[-1]) for file in files}
        files = OrderedDict(sorted(files.items(), key=lambda x: x[1]))
        result = []
        start = 0
        for name, end in files.items():
            intersection = pd.np.intersect1d(iterations, pd.np.arange(start, end))
            if len(intersection) > 0:
                result.append((name, intersection))
            start = end
        return OrderedDict(result)

    def _slice_file(self, dumped_file, iterations_to_load, variables):
        iterations = dumped_file['iteration']
        if len(iterations) > 0:
            elements_to_load = pd.np.array([pd.np.isin(it, iterations_to_load) for it in iterations])
            res = OrderedDict()
            for variable in ['iteration', *variables]:
                if variable in dumped_file:
                    res[variable] = pd.np.array(dumped_file[variable])[elements_to_load]
        else:
            res = None
        return res

    def _concat(self, results, variables):
        res = {key: [] for key in [*variables, 'iteration']}
        for chunk in results:
            if chunk is not None:
                for key, values in res.items():
                    if key in chunk:
                        values.extend(chunk[key])
        return res

    def _fix_length(self, chunk):
        max_len = max([len(value) for value in chunk.values()])
        for value in chunk.values():
            if len(value) < max_len:
                value.extend([pd.np.nan] * (max_len - len(value)))

    def _filter_configs(self, config=None, alias=None):
        result = None
        if config is None and alias is None:
            raise ValueError('At least one of parameters config and alias must be not None')
        if config is not None:
            result = self.configs.subset(config, by_alias=False)
        else:
            result = self.configs.subset(alias, by_alias=True)
        return result


    def load(self, names=None, repetitions=None, folds=None, variables=None, iterations=None,
             configs=None, aliases=None, use_alias=False):
        """ Load results as pandas.DataFrame.

        Parameters
        ----------
        names : str, list or None
            names of units (pipleines and functions) to load
        repetitions : int, list or None
            numbers of repetitions to load
        folds : int, list or None
            split of dataset
        variables : str, list or None
            names of variables to load
        iterations : int, list or None
            iterations to load
        configs, aliases : dict, Config, Option, Grid or None
            configs to load
        use_alias : bool
            if True, the resulting DataFrame will have one column with alias, else it will
            have column for each option in grid

        Returns
        -------
        pandas.DataFrame or dict
            will have columns: iteration, repetition, name (of pipeline/function)
            and column for config. Also it will have column for each variable of pipeline
            and output of the function that was saved as a result of the research.

        **How to perform slicing**
            Method `load` with default parameters will create pandas.DataFrame with all dumped
            parameters. To specify subset of results one can define names of pipelines/functions,
            produced variables/outputs of them, repetitions, iterations and configs. For example,
            we have the following research:

            ```
            grid = Option('layout', ['cna', 'can', 'acn']) * Option('model', [VGG7, VGG16])

            research = (Research()
            .add_pipeline(train_ppl, variables='loss', name='train')
            .add_pipeline(test_ppl, name='test', execute='%100', run=True, import_from='train')
            .add_function(accuracy, returns='accuracy', name='test_accuracy',
                      execute='%100', pipeline='test')
            .add_grid(grid))

            research.run(n_reps=2, n_iters=10000)
            ```
            The code
            ```
            Results(research=research).load(repetitions=0, iterations=np.arange(5000, 10000),
                                            variables='accuracy', names='test_accuracy',
                                            configs=Option('layout', ['cna', 'can']))
            ```
            will load output of ``accuracy`` function at the first repetitions for configs
            that contain layout 'cna' or 'can' for iterations starting with 5000.
            The resulting dataframe will have columns 'repetition', 'iteration', 'name',
            'accuracy', 'layout', 'model'. One can get the same in the follwing way:
            ```
            results = Results(research=research).load()
            results = results[(results.repetition == 0) & (results.iterations >= 5000) &
                              (results.name == 'test_accuracy') & results.layout.isin(['cna', 'can'])]
            ```
        """
        self.configs = self.research.grid_config
        if configs is None and aliases is None:
            self.configs = list(self.configs.gen_configs())
        elif configs is not None:
            self.configs = self._filter_configs(config=configs)
        else:
            self.configs = self._filter_configs(alias=aliases)

        if names is None:
            names = list(self.research.executables.keys())

        if repetitions is None:
            repetitions = list(range(self.research.n_reps))

        if folds is None:
            folds = list(range(self.research.n_splits)) if self.research.n_splits is not None else [None]

        if variables is None:
            variables = [variable for unit in self.research.executables.values() for variable in unit.variables]

        if iterations is None:
            iterations = list(range(self.research.n_iters))

        self.names = self._get_list(names)
        self.repetitions = self._get_list(repetitions)
        self.variables = self._get_list(variables)
        self.iterations = self._get_list(iterations)
        self.folds = self._get_list(folds)

        all_results = []

        for config_alias in self.configs:
            alias = config_alias.alias(as_string=False)
            alias_str = config_alias.alias(as_string=True)
            for repetition in self.repetitions:
                for cv_split in self.folds:
                    for unit in self.names:
                        path = os.path.join(self.path, 'results', alias_str, str(repetition))
                        cv_folder = 'cv_'+str(cv_split) if cv_split is not None else ''
                        files = glob.glob(os.path.join(glob.escape(path), cv_folder, unit + '_[0-9]*'))
                        files = self._sort_files(files, self.iterations)
                        if len(files) != 0:
                            res = []
                            for filename, iterations_to_load in files.items():
                                with open(filename, 'rb') as file:
                                    res.append(self._slice_file(dill.load(file), iterations_to_load, self.variables))
                            res = self._concat(res, self.variables)
                            self._fix_length(res)
                            if '_dummy' not in alias:
                                if use_alias:
                                    res['config'] = alias_str
                                else:
                                    res.update(alias)
                            if cv_split is not None:
                                res['cv_split'] = cv_split
                            all_results.append(
                                pd.DataFrame({
                                    'repetition': repetition,
                                    'name': unit,
                                    **res
                                })
                                )
        return pd.concat(all_results) if len(all_results) > 0 else pd.DataFrame(None)
