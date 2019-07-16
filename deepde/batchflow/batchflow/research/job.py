""" Classes Job and Experiment. """

from collections import OrderedDict

from .. import inbatch_parallel

class Job:
    """ Contains one job. """
    def __init__(self, executable_units, n_iters, repetition, configs, folds, branches, name):
        """
        Parameters
        ----------
        config : dict or Config
            config of experiment
        """
        self.experiments = []
        self.executable_units = executable_units
        self.n_iters = n_iters
        self.configs = configs
        self.repetition = repetition
        self.folds = folds
        self.branches = branches
        self.name = name
        self.worker_config = {}

        self.exceptions = []
        self.stopped = []

    def init(self, worker_config, gpu_configs):
        """ Create experiments. """
        self.worker_config = worker_config

        for index, config in enumerate(self.configs):
            if isinstance(self.branches, list):
                branch_config = self.branches[index]
            else:
                branch_config = dict()
            units = OrderedDict()
            for name, unit in self.executable_units.items():
                unit = unit.get_copy()
                unit.reset_iter()
                unit.cv_split = self.folds[index]
                if unit.pipeline is not None:
                    import_config = {key: units[value].pipeline for key, value in unit.kwargs.items()}
                    unit.set_dataset()
                else:
                    import_config = dict()
                unit.set_config(config, {**branch_config, **gpu_configs[index]}, worker_config, import_config)
                unit.repetition = self.repetition[index]
                unit.index = index
                unit.create_folder(self.name)
                units[name] = unit

            self.experiments.append(units)
            self.exceptions.append(None)
        self.clear_stopped()

    def clear_stopped(self):
        """ Clear list of stopped experiments for the current iteration """
        self.stopped = [False for _ in range(len(self.experiments))]

    def get_description(self):
        """ Get description of job. """
        if isinstance(self.branches, list):
            description = '\n'.join([str({**config.alias(), **_config, **self.worker_config})
                                     for config, _config in zip(self.configs, self.branches)])
        else:
            description = '\n'.join([str({**config.alias(), **self.worker_config})
                                     for config in self.configs])
        return description

    def parallel_execute_for(self, iteration, name, actions):
        """ Parallel execution of pipeline 'name' """
        run = [action['run'] for action in actions if action is not None][0]
        if run:
            self.executable_units[name].reset_root_iter()
            while True:
                try:
                    batch = self.executable_units[name].next_batch_root()
                    exceptions = self._parallel_run(iteration, name, batch, actions) #pylint:disable=assignment-from-no-return
                except StopIteration:
                    break
        else:
            try:
                batch = self.executable_units[name].next_batch_root()
            except StopIteration as e:
                exceptions = [e] * len(self.experiments)
            else:
                exceptions = self._parallel_run(iteration, name, batch, actions) #pylint:disable=assignment-from-no-return
        self.put_all_results(iteration, name, actions)
        return exceptions

    def update_exceptions(self, exceptions):
        """ Update exceptions with new from current iteration """
        for i, exception in enumerate(exceptions):
            if exception is not None:
                self.exceptions[i] = exception

    @inbatch_parallel(init='_parallel_init_run', post='_parallel_post')
    def _parallel_run(self, item, execute, iteration, name, batch, actions):
        _ = name, actions
        if execute is not None:
            item.execute_for(batch, iteration)

    def _parallel_init_run(self, iteration, name, batch, actions):
        _ = iteration, batch
        #to_run = self._experiments_to_run(iteration, name)
        return [[experiment[name], execute] for experiment, execute in zip(self.experiments, actions)]

    def _parallel_post(self, results, *args, **kwargs):
        _ = args, kwargs
        return results

    @inbatch_parallel(init='_parallel_init_call', post='_parallel_post')
    def parallel_call(self, item, execute, iteration, name, actions):
        """ Parallel call of the unit 'name' """
        _ = actions
        if execute is not None:
            item[name](iteration, item, *item[name].args, **item[name].kwargs)

    def _parallel_init_call(self, iteration, name, actions):
        _ = iteration, name
        return [[experiment, execute] for experiment, execute in zip(self.experiments, actions)]

    def put_all_results(self, iteration, name, actions):
        """ Add values of pipeline variables to results """
        for experiment, execute in zip(self.experiments, actions):
            if execute is not None:
                experiment[name].put_result(iteration)

    def get_actions(self, iteration, name, action='execute'):
        """ Experiments that should be executed """
        res = []
        for idx, experiment in enumerate(self.experiments):
            if experiment[name].action_iteration(iteration, self.n_iters, action) and self.exceptions[idx] is None:
                res.append(experiment[name].action)
            elif (self.stopped[idx]) and -1 in getattr(experiment[name], action):
                res.append(experiment[name].action)
            else:
                res.append(None)
        return res

    def all_stopped(self):
        """ Does all experiments finished """
        res = True
        for exception in self.exceptions:
            res = isinstance(exception, StopIteration)
        return res

    def alive_experiments(self):
        """ Get number of alive experiments """
        return len([item for item in self.exceptions if item is None])
