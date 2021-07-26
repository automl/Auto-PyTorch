from hpbandster.core.result import Result
from hpbandster.core.dispatcher import Job
import copy
import time

def run_with_time(self, runtime=1, n_iterations=float("inf"), min_n_workers=1, iteration_kwargs = {},):
    """
        custom run method of Master in hpbandster submodule

        Parameters:
        -----------
        runtime: int
            time for this run in seconds
        n_iterations: int
            maximum number of iterations
        min_n_workers: int
            minimum number of workers before starting the run
    """

    self.wait_for_workers(min_n_workers)
    
    iteration_kwargs.update({'result_logger': self.result_logger})

    if self.time_ref is None:
        self.time_ref = time.time()
        self.config['time_ref'] = self.time_ref
    
        self.logger.info('HBMASTER: starting run at %s'%(str(self.time_ref)))

    self.thread_cond.acquire()

    start_time = time.time()
    while True:
        self._queue_wait()

        if (runtime < time.time() - start_time):
            # timelimit reached -> finish
            self.logger.info('HBMASTER: Timelimit reached: wait for remaining %i jobs'%self.num_running_jobs)
            break

        next_run = None
        # find a new run to schedule
        for i in self.active_iterations():
            next_run = self.iterations[i].get_next_run()
            if not next_run is None: break

        if not next_run is None:
            self.logger.debug('HBMASTER: schedule new run for iteration %i'%i)
            self._submit_job(*next_run)												# Submits configs for one iteration of one SH run
            continue
        elif n_iterations > 0:
            self.iterations.append(self.get_next_iteration(len(self.iterations), iteration_kwargs))				# Gets configs for one whole SH run
            n_iterations -= 1
            continue

        # at this point there is no imediate run that can be scheduled,
        # so wait for some job to finish if there are active iterations
        if self.active_iterations():
            self.thread_cond.wait()
        else:
            break

    # clean up / cancel remaining iteration runs
    next_run = True
    n_canceled = 0
    while next_run is not None:
        next_run = None
        for i in self.active_iterations():
            next_run = self.iterations[i].get_next_run()
            if not next_run is None: 
                config_id, config, budget = next_run
                job = Job(config_id, config=config, budget=budget, working_directory=self.working_directory)
                self.iterations[job.id[0]].register_result(job) # register dummy job - will be interpreted as canceled job
                n_canceled += 1
                break
    
    self.logger.info('HBMASTER: Canceled %i remaining runs'%n_canceled)

    # wait for remaining jobs
    while self.num_running_jobs > 0:
        self.thread_cond.wait(60)
        self.logger.info('HBMASTER: Job finished: wait for remaining %i jobs'%self.num_running_jobs)

    self.thread_cond.release()
    
    for i in self.warmstart_iteration:
        i.fix_timestamps(self.time_ref)
        
    ws_data = [i.data for i in self.warmstart_iteration]
    
    return Result([copy.deepcopy(i.data) for i in self.iterations] + ws_data, self.config)
