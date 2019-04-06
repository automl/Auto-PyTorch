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
    n_iterations -= len(self.iterations)  # in the case of a initial design iteration
    kill = False
    while True:
        if (not kill and runtime < time.time() - start_time):
            # wait for running jobs and finish
            kill = True
            self.logger.info('HBMASTER: Timelimit reached: wait for remaining %i jobs'%self.num_running_jobs)
                

        self._queue_wait()
        
        next_run = None
        # find a new run to schedule
        for i in self.active_iterations():
            next_run = self.iterations[i].get_next_run()
            if not next_run is None: break

        if next_run is not None:
            if kill:
                # register new run as finished - this will be interpreted as a crashed job
                config_id, config, budget = next_run
                job = Job(config_id, config=config, budget=budget, working_directory=self.working_directory)
                self.iterations[job.id[0] - self.iterations[0].HPB_iter].register_result(job)
            else:
                self.logger.debug('HBMASTER: schedule new run for iteration %i'%i)
                self._submit_job(*next_run)
            continue
        elif not kill and n_iterations > 0:
            next_HPB_iter = len(self.iterations) + (self.iterations[0].HPB_iter if len(self.iterations) > 0 else 0)
            self.iterations.append(self.get_next_iteration(next_HPB_iter, iteration_kwargs))
            n_iterations -= 1
            continue

        # at this point there is no imediate run that can be scheduled,
        # so wait for some job to finish if there are active iterations
        if self.active_iterations():
            self.thread_cond.wait()
        else:
            break

    self.thread_cond.release()
    
    for i in self.warmstart_iteration:
        i.fix_timestamps(self.time_ref)
        
    ws_data = [i.data for i in self.warmstart_iteration]
    
    return Result([copy.deepcopy(i.data) for i in self.iterations] + ws_data, self.config)
