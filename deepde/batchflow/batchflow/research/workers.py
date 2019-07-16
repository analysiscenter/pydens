""" Workers for research. """

import os

from .distributor import Worker, Signal

class PipelineWorker(Worker):
    """ Worker that run pipelines. """

    def init(self):
        """ Run before job execution. """
        i, job = self.job
        n_branches = len(job.configs)

        if len(self.gpu) <= 1:
            self.gpu_configs = [dict(device='gpu:0') for i in range(n_branches)]
        else:
            self.gpu_configs = [dict(device='gpu:'+str(i)) for i in range(n_branches)]

        job.init(self.worker_config, self.gpu_configs)

        description = job.get_description()
        self.log_info('Job {} has the following configs:\n{}'.format(i, description), filename=self.logfile)

    def post(self):
        """ Run after job execution. """
        pass #pylint:disable=unnecessary-pass

    def _execute_on_root(self, base_unit, iteration):
        _, job = self.job
        return base_unit.action_iteration(iteration, job.n_iters) or (-1 in base_unit.execute) and job.all_stopped()

    def run_job(self):
        """ Job execution. """
        idx_job, job = self.job

        iteration = 0
        self.finished_iterations = iteration

        while (job.n_iters is None or iteration < job.n_iters) and job.alive_experiments() > 0:
            job.clear_stopped()
            for unit_name, base_unit in job.executable_units.items():
                exec_actions = job.get_actions(iteration, unit_name) # for each experiment is None if experiment mustn't
                                                                     # be exuted for that iteration and dict else
                # execute units
                messages = []
                exceptions = [None] * len(job.experiments)
                if base_unit.root_pipeline is not None:
                    if sum([item is not None for item in exec_actions]) > 0:
                        for i, action in enumerate(exec_actions):
                            if action is not None:
                                messages.append("J {} [{}] I {}: execute '{}' [{}]"
                                                .format(idx_job, os.getpid(), iteration+1, unit_name, i))
                        exceptions = job.parallel_execute_for(iteration, unit_name, exec_actions)
                elif base_unit.on_root and self._execute_on_root(base_unit, iteration):
                    try:
                        for i, action in enumerate(exec_actions):
                            if action is not None:
                                messages.append("J {} [{}] I {}: on root '{}' [{}]"
                                                .format(idx_job, os.getpid(), iteration+1, unit_name, i))
                        base_unit(iteration, job.experiments, *base_unit.args, **base_unit.kwargs)
                    except Exception as e: #pylint:disable=broad-except
                        exceptions = [e] * len(job.experiments)
                else:
                    for i, action in enumerate(exec_actions):
                        if action is not None:
                            messages.append("J {} [{}] I {}: execute '{}' [{}]"
                                            .format(idx_job, os.getpid(), iteration+1, unit_name, i))
                    exceptions = job.parallel_call(iteration, unit_name, exec_actions)

                # select units that raise exceptions on that iterartion
                for i, exception in enumerate(exceptions):
                    if exception is not None:
                        message = ("J {} [{}] I {}: '{}' [{}]: exception {}"
                                   .format(idx_job, os.getpid(), iteration+1, unit_name, i, repr(exception)))
                        self.log_info(message, filename=self.logfile)
                        job.stopped[i] = True

                # dump results
                dump_actions = job.get_actions(iteration, unit_name, action='dump')
                for i, experiment in enumerate(job.experiments):
                    if dump_actions[i] is not None:
                        messages.append("J {} [{}] I {}: dump '{}' [{}]"
                                        .format(idx_job, os.getpid(), iteration+1, unit_name, i))
                        experiment[unit_name].dump_result(iteration+1, unit_name)

                if base_unit.logging:
                    for message in messages:
                        self.log_info(message, filename=self.logfile)
                job.update_exceptions(exceptions)
                signal = Signal(self.worker, idx_job, iteration, job.n_iters, self.trial,
                                False, job.exceptions, exec_actions, dump_actions)
                self.feedback_queue.put(signal)
            iteration += 1
            self.finished_iterations = iteration
