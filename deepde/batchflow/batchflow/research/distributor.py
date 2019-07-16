""" Classes for multiprocess job running. """

import os
import logging
from queue import Empty
import multiprocess as mp
from tqdm import tqdm
import psutil

class Distributor:
    """ Distributor of jobs between workers. """
    def __init__(self, workers, gpu, worker_class=None, timeout=5, trials=3):
        """
        Parameters
        ----------
        workers : int or list of Worker instances

        worker_class : Worker subclass or None
        """
        self.workers = workers
        self.worker_class = worker_class
        self.gpu = gpu
        self.timeout = timeout
        self.trials = trials
        self.logfile = None
        self.errorfile = None
        self.results = None
        self.finished_jobs = None
        self.answers = None
        self.queue = None

    def _jobs_to_queue(self, jobs):
        queue = mp.JoinableQueue()
        for idx, job in enumerate(jobs):
            queue.put((idx, job))
        for _ in range(self.workers):
            queue.put(None)
        return queue

    @classmethod
    def log_info(cls, message, filename):
        """ Write message into log. """
        logging.basicConfig(format='%(levelname)-8s [%(asctime)s] %(message)s', filename=filename, level=logging.INFO)
        logging.info(message)

    @classmethod
    def log_error(cls, obj, filename):
        """ Write error message into log. """
        logging.basicConfig(format='%(levelname)-8s [%(asctime)s] %(message)s', filename=filename, level=logging.INFO)
        logging.error(obj, exc_info=True)

    def _get_worker_gpu(self, n_workers, index):
        if len(self.gpu) == 1:
            gpu = [self.gpu[0]]
        else:
            length = len(self.gpu) // n_workers
            start = index * length
            end = start + length
            gpu = self.gpu[start:end]
        return gpu

    def run(self, jobs, dirname, n_jobs, n_iters, logfile=None, errorfile=None, bar=False, *args, **kwargs):
        """ Run disributor and workers.

        Parameters
        ----------
        jobs : iterable

        dirname : str

        logfile : str (default: 'research.log')

        errorfile : str (default: 'errors.log')

        bar : bool or callable

        args, kwargs
            will be used in worker
        """
        if isinstance(bar, bool):
            bar = tqdm if bar else None

        self.logfile = logfile or 'research.log'
        self.errorfile = errorfile or 'errors.log'

        self.logfile = os.path.join(dirname, self.logfile)
        self.errorfile = os.path.join(dirname, self.errorfile)

        kwargs['logfile'] = self.logfile
        kwargs['errorfile'] = self.errorfile

        self.log_info('Distributor [id:{}] is preparing workers'.format(os.getpid()), filename=self.logfile)

        if isinstance(self.workers, int):
            workers = [self.worker_class(
                gpu=self._get_worker_gpu(self.workers, i),
                worker_name=i,
                timeout=self.timeout,
                trials=self.trials,
                *args, **kwargs
                )
                       for i in range(self.workers)]
        else:
            workers = [
                self.worker_class(gpu=self._get_worker_gpu(len(self.workers), i), worker_name=i, config=config,
                                  timeout=self.timeout, trials=self.trials, *args, **kwargs)
                for i, config in enumerate(self.workers)
            ]
        try:
            self.log_info('Create queue of jobs', filename=self.logfile)
            self.queue = self._jobs_to_queue(jobs)
            self.results = mp.JoinableQueue()
        except Exception as exception: #pylint:disable=broad-except
            logging.error(exception, exc_info=True)
        else:
            if len(workers) > 1:
                msg = 'Run {} workers'
            else:
                msg = 'Run {} worker'
            self.log_info(msg.format(len(workers)), filename=self.logfile)
            for worker in workers:
                worker.log_info = self.log_info
                worker.log_error = self.log_error
                try:
                    mp.Process(target=worker, args=(self.queue, self.results)).start()
                except Exception as exception: #pylint:disable=broad-except
                    logging.error(exception, exc_info=True)

            self.answers = [0 for _ in range(n_jobs)]
            self.finished_jobs = []


            if bar is not None:
                if n_iters is not None:
                    print("Distributor has {} jobs with {} iterations. Totally: {}"
                          .format(n_jobs, n_iters, n_jobs*n_iters))
                    with bar(total=n_jobs*n_iters) as progress:
                        while True:
                            position = self._get_position()
                            progress.n = position
                            progress.refresh()
                            if len(self.finished_jobs) == n_jobs:
                                break
                else:
                    print("Distributor has {} jobs"
                          .format(n_jobs))
                    with bar(total=n_jobs) as progress:
                        while True:
                            answer = self.results.get()
                            if answer.done:
                                self.finished_jobs.append(answer.job)
                            position = len(self.finished_jobs)
                            progress.n = position
                            progress.refresh()
                            if len(self.finished_jobs) == n_jobs:
                                break
            else:
                self.queue.join()
        self.log_info('All workers have finished the work', filename=self.logfile)
        logging.shutdown()



    def _get_position(self, fixed_iterations=True):
        answer = self.results.get()
        if answer.done:
            self.finished_jobs.append(answer.job)
        if fixed_iterations:
            if answer.done:
                self.answers[answer.job] = answer.n_iters
            else:
                self.answers[answer.job] = answer.iteration+1
            return sum(self.answers)

        self.answers[answer.job] += 1
        return sum(self.answers)

class Worker:
    """ Worker that creates subprocess to execute job.
    Worker get queue of jobs, pop one job and execute it in subprocess. That subprocess
    call init, run_job and post class methods.
    """
    def __init__(self, gpu, worker_name=None, logfile=None, errorfile=None,
                 config=None, timeout=5, trials=2, *args, **kwargs):
        """
        Parameters
        ----------
        worker_name : str or int

        logfile : str (default: 'research.log')

        errorfile : str (default: 'errors.log')

        config : dict or str
            additional config for pipelines in worker
        args, kwargs
            will be used in init, post and run_job
        """
        self.job = None
        self.worker_config = config or dict()
        self.args = args
        self.kwargs = kwargs
        self.gpu = gpu
        self.timeout = timeout
        self.trials = trials
        self.gpu_configs = None
        self.finished_iterations = None
        self.queue = None
        self.feedback_queue = None
        self.trial = 3
        self.worker = None

        if isinstance(worker_name, int):
            self.name = "Worker " + str(worker_name)
        elif worker_name is None:
            self.name = 'Worker'
        else:
            self.name = worker_name
        self.logfile = logfile or 'research.log'
        self.errorfile = errorfile or 'errors.log'

    def set_args_kwargs(self, args, kwargs):
        """
        Parameters
        ----------
        args, kwargs
            will be used in init, post and run_job
        """
        if 'logfile' in kwargs:
            self.logfile = kwargs['logfile']
        if 'errorfile' in kwargs:
            self.errorfile = kwargs['errorfile']
        self.logfile = self.logfile or 'research.log'
        self.errorfile = self.errorfile or 'errors.log'

        self.args = args
        self.kwargs = kwargs

    def init(self):
        """ Run before run_job. """
        pass #pylint:disable=unnecessary-pass

    def post(self):
        """ Run after run_job. """
        pass #pylint:disable=unnecessary-pass

    def run_job(self):
        """ Main part of the worker. """
        pass #pylint:disable=unnecessary-pass


    def __call__(self, queue, results):
        """ Run worker.

        Parameters
        ----------
        queue : multiprocessing.Queue
            queue of jobs for worker
        results : multiprocessing.Queue
            queue for feedback
        """
        _gpu = 'default' if len(self.gpu) == 0 else self.gpu
        self.log_info('Start {} [id:{}] (gpu: {})'.format(self.name, os.getpid(), _gpu), filename=self.logfile)

        if len(self.gpu) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in self.gpu])

        try:
            job = queue.get()
        except Exception as exception: #pylint:disable=broad-except
            self.log_error(exception, filename=self.errorfile)
        else:
            while job is not None:
                try:
                    finished = False
                    self.log_info(self.name + ' is creating process for Job ' + str(job[0]), filename=self.logfile)
                    for trial in range(self.trials):
                        sub_queue = mp.JoinableQueue()
                        sub_queue.put(job)
                        feedback_queue = mp.JoinableQueue()

                        worker = mp.Process(target=self._run_job, args=(sub_queue, feedback_queue, self.name, trial))
                        worker.start()
                        pid = feedback_queue.get()
                        silence = 0
                        default_signal = Signal(self.name, job[0], 0, job[1].n_iters, trial, False, None)

                        while True:
                            try:
                                signal = feedback_queue.get(timeout=1)
                            except Empty:
                                signal = None
                                silence += 1
                            if signal is None and silence / 60 > self.timeout:
                                p = psutil.Process(pid)
                                p.terminate()
                                message = 'Job {} [{}] failed in {}'.format(job[0], pid, self.name)
                                self.log_info(message, filename=self.logfile)
                                default_signal.exception = TimeoutError(message)
                                results.put(default_signal)
                                break
                            elif signal is not None and signal.done:
                                finished = True
                                default_signal = signal
                                break
                            elif signal is not None:
                                default_signal = signal
                                results.put(default_signal)
                                silence = 0
                        if finished:
                            break
                except Exception as exception: #pylint:disable=broad-except
                    self.log_error(exception, filename=self.errorfile)
                    default_signal.exception = exception
                    results.put(default_signal)
                if default_signal.done:
                    results.put(default_signal)
                else:
                    default_signal.exception = RuntimeError('Job {} [{}] failed {} times in {}'
                                                            .format(job[0], pid, self.trials, self.name))
                    results.put(default_signal)
                queue.task_done()
                job = queue.get()
            queue.task_done()


    def _run_job(self, queue, feedback_queue, worker, trial):
        exception = None
        try:
            self.feedback_queue = feedback_queue
            self.worker = worker
            self.trial = trial

            feedback_queue.put(os.getpid())
            self.job = queue.get()

            self.log_info(
                'Job {} was started in subprocess [id:{}] by {}'.format(self.job[0], os.getpid(), self.name),
                filename=self.logfile
            )
            self.init()
            self.run_job()
            self.post()
        except Exception as e: #pylint:disable=broad-except
            exception = e
            self.log_error(exception, filename=self.errorfile)
        self.log_info('Job {} [{}] was finished by {}'.format(self.job[0], os.getpid(), self.name),
                      filename=self.logfile)
        signal = Signal(self.worker, self.job[0], self.finished_iterations, self.job[1].n_iters,
                        self.trial, True, [exception]*len(self.job[1].experiments))
        self.feedback_queue.put(signal)
        queue.task_done()

    @classmethod
    def log_info(cls, *args, **kwargs):
        """ Write message into log """
        pass #pylint:disable=unnecessary-pass

    @classmethod
    def log_error(cls, *args, **kwargs):
        """ Write error message into log """
        pass #pylint:disable=unnecessary-pass

class Signal:
    """ Class for feedback from jobs and workers """
    def __init__(self, worker, job, iteration, n_iters, trial, done, exception, exec_actions=None, dump_actions=None):
        self.worker = worker
        self.job = job
        self.iteration = iteration
        self.n_iters = n_iters
        self.trial = trial
        self.done = done
        self.exception = exception
        self.exec_actions = exec_actions
        self.dump_actions = dump_actions

    def __repr__(self):
        return str(self.__dict__)
