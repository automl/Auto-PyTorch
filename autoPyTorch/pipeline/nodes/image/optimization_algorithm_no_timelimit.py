import numpy as np
import os
import time
import shutil
import netifaces
import traceback
import logging
import itertools
import random

import autoPyTorch.utils.thread_read_write as thread_read_write
import datetime

from hpbandster.core.nameserver import NameServer, nic_name_to_host
from hpbandster.core.result import (json_result_logger,
                                    logged_results_to_HBS_result)

from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool

from autoPyTorch.core.hpbandster_extensions.bohb_ext import BOHBExt
from autoPyTorch.core.hpbandster_extensions.hyperband_ext import HyperBandExt
from autoPyTorch.core.worker_no_timelimit import ModuleWorkerNoTimeLimit

from autoPyTorch.components.training.image.budget_types import BudgetTypeTime, BudgetTypeEpochs
import copy

from autoPyTorch.utils.modify_config_space import remove_constant_hyperparameter

from autoPyTorch.utils.loggers import combined_logger, bohb_logger, tensorboard_logger

import pprint

tensorboard_logger_configured = False

class OptimizationAlgorithmNoTimeLimit(SubPipelineNode):
    def __init__(self, optimization_pipeline_nodes):
        """OptimizationAlgorithm pipeline node.
        It will run either the optimization algorithm (BOHB, Hyperband - defined in config) or start workers
        Each worker will run the provided optimization_pipeline and will return the output 
        of the pipeline_result_node to the optimization algorithm

        Train:
        The optimization_pipeline will get the following inputs:
        {hyperparameter_config, pipeline_config, X_train, Y_train, X_valid, Y_valid, budget, budget_type}
        The pipeline_result_node has to provide the following outputs:
        - 'loss': the optimization value (minimize)
        - 'info': dict containing info for the respective training process

        Predict:
        The optimization_pipeline will get the following inputs:
        {pipeline_config, X}
        The pipeline_result_node has to provide the following outputs:
        - 'Y': result of prediction for 'X'
        Note: predict will not call the optimization algorithm
        
        Arguments:
            optimization_pipeline {Pipeline} -- pipeline that will be optimized (hyperparamter)
            pipeline_result_node {PipelineNode} -- pipeline node that provides the results of the optimization_pieline
        """

        super(OptimizationAlgorithmNoTimeLimit, self).__init__(optimization_pipeline_nodes)

        self.algorithms = dict()
        self.algorithms["bohb"] = BOHBExt
        self.algorithms["hyperband"] = HyperBandExt

        self.logger = logging.getLogger('autonet')

        self.n_datasets=1

    def fit(self, pipeline_config, X_train, Y_train, X_valid, Y_valid, refit=None):
        res = None

        config_space = self.pipeline.get_hyperparameter_search_space(**pipeline_config)
        
        config_space, constants = remove_constant_hyperparameter(config_space)
        config_space.seed(pipeline_config['random_seed'])

        self.n_datasets = X_train.shape[0] if X_train.shape[0]<10 else 1

        #Get number of budgets
        max_budget = pipeline_config["max_budget"]
        min_budget = pipeline_config["min_budget"]
        eta = pipeline_config["eta"]
        max_SH_iter = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
        budgets = max_budget * np.power(eta, -np.linspace(max_SH_iter-1, 0, max_SH_iter))
        n_budgets = len(budgets)

        # Get permutations
        self.permutations = self.get_permutations(n_budgets)

        self.logger.debug('BOHB-ConfigSpace:\n' + str(config_space))
        self.logger.debug('Constant Hyperparameter:\n' + str(pprint.pformat(constants)))

        run_id, task_id = pipeline_config['run_id'], pipeline_config['task_id']


        global tensorboard_logger_configured
        if pipeline_config['use_tensorboard_logger'] and not tensorboard_logger_configured:            
            import tensorboard_logger as tl
            directory = os.path.join(pipeline_config['result_logger_dir'], "worker_logs_" + str(task_id))
            os.makedirs(directory, exist_ok=True)
            tl.configure(directory, flush_secs=60)
            tensorboard_logger_configured = True

        if (refit is not None):
            return self.run_refit(pipeline_config, refit, constants, X_train, Y_train, X_valid, Y_valid)

        try:
            ns_credentials_dir, tmp_models_dir, network_interface_name = self.prepare_environment(pipeline_config)

            # start nameserver if not on cluster or on master node in cluster
            if task_id in [1, -1]:
                NS = self.get_nameserver(run_id, task_id, ns_credentials_dir, network_interface_name)
                ns_host, ns_port = NS.start()

            self.run_worker(pipeline_config=pipeline_config, run_id=run_id, task_id=task_id, ns_credentials_dir=ns_credentials_dir,
                network_interface_name=network_interface_name, X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid,
                constant_hyperparameter=constants)

            # start BOHB if not on cluster or on master node in cluster
            if task_id in [1, -1]:
                self.run_optimization_algorithm(pipeline_config, config_space, constants, run_id, ns_host, ns_port, NS, task_id)
            
            res = self.parse_results(pipeline_config["result_logger_dir"])

        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            self.clean_up(pipeline_config, ns_credentials_dir, tmp_models_dir)

        if (res):
            return {'loss': res[0], 'optimized_hyperparameter_config': res[1], 'budget': res[2], 'info': dict()}
        else:
            return {'optimized_hyperparameter_config': dict(), 'budget': 0, 'loss': float('inf'), 'info': dict()}

    def predict(self, pipeline_config, X):
        return self.sub_pipeline.predict_pipeline(pipeline_config=pipeline_config, X=X)

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("run_id", default="0", type=str, info="Unique id for each run."),
            ConfigOption("task_id", default=-1, type=int, info="ID for each worker, if you run AutoNet on a cluster. Set to -1, if you run it locally. "),
            ConfigOption("algorithm", default="bohb", type=str, choices=list(self.algorithms.keys())),
            ConfigOption("budget_type", default="time", type=str, choices=['time', 'epochs']),
            ConfigOption("min_budget", default=lambda c: 120 if c['budget_type'] == 'time' else 5, type=float, depends=True, info="Min budget for fitting configurations."),
            ConfigOption("max_budget", default=lambda c: 6000 if c['budget_type'] == 'time' else 150, type=float, depends=True, info="Max budget for fitting configurations."),
            ConfigOption("max_runtime", 
                default=lambda c: ((-int(np.log(c["min_budget"] / c["max_budget"]) / np.log(c["eta"])) + 1) * c["max_budget"])
                        if c["budget_type"] == "time" else float("inf"),
                type=float, depends=True, info="Total time for the run."),
            ConfigOption("num_iterations", 
                default=lambda c:  (-int(np.log(c["min_budget"] / c["max_budget"]) / np.log(c["eta"])) + 1)
                        if c["budget_type"] == "epochs" else float("inf"),
                type=float, depends=True, info="Number of successive halving iterations"),
            ConfigOption("eta", default=3, type=float, info='eta parameter of Hyperband.'),
            ConfigOption("min_workers", default=1, type=int),
            ConfigOption("working_dir", default=".", type="directory"),
            ConfigOption("network_interface_name", default=self.get_default_network_interface_name(), type=str),
            ConfigOption("memory_limit_mb", default=1000000, type=int),
            ConfigOption("result_logger_dir", default=".", type="directory"),
            ConfigOption("use_tensorboard_logger", default=False, type=to_bool),
            ConfigOption("keep_only_incumbent_checkpoints", default=True, type=to_bool),
            ConfigOption("global_results_dir", default=None, type='directory'),
        ]
        return options

    def get_default_network_interface_name(self):
        try:
            return netifaces.gateways()['default'][netifaces.AF_INET][1]
        except:
            return 'lo'

    def prepare_environment(self, pipeline_config):
        if not os.path.exists(pipeline_config["working_dir"]) and pipeline_config['task_id'] in [1, -1]:
            try:
                os.mkdir(pipeline_config["working_dir"])
            except:
                pass
        tmp_models_dir = os.path.join(pipeline_config["working_dir"], "tmp_models_" + str(pipeline_config['run_id']))
        ns_credentials_dir = os.path.abspath(os.path.join(pipeline_config["working_dir"], "ns_credentials_" + str(pipeline_config['run_id'])))
        network_interface_name = pipeline_config["network_interface_name"] or (netifaces.interfaces()[1] if len(netifaces.interfaces()) > 1 else "lo")
        
        if os.path.exists(tmp_models_dir) and pipeline_config['task_id'] in [1, -1]:
            shutil.rmtree(tmp_models_dir)
        if os.path.exists(ns_credentials_dir) and pipeline_config['task_id'] in [1, -1]:
            shutil.rmtree(ns_credentials_dir)
        return ns_credentials_dir, tmp_models_dir, network_interface_name

    def clean_up(self, pipeline_config, tmp_models_dir, ns_credentials_dir):
        if pipeline_config['task_id'] in [1, -1]:
            # Delete temporary files
            if os.path.exists(tmp_models_dir):
                shutil.rmtree(tmp_models_dir)
            if os.path.exists(ns_credentials_dir):
                shutil.rmtree(ns_credentials_dir)

    def get_nameserver(self, run_id, task_id, ns_credentials_dir, network_interface_name):
        if not os.path.isdir(ns_credentials_dir):
            try:
                os.mkdir(ns_credentials_dir)
            except:
                pass
        return NameServer(run_id=run_id, nic_name=network_interface_name, working_directory=ns_credentials_dir)
    
    def get_optimization_algorithm_instance(self, config_space, run_id, pipeline_config, ns_host, ns_port, result_logger, previous_result=None):
        optimization_algorithm = self.algorithms[pipeline_config["algorithm"]]

        if pipeline_config["algorithm"]=="bohb_multi_kde":
            hb = optimization_algorithm(configspace=config_space, run_id = run_id,
                                        eta=pipeline_config["eta"], min_budget=pipeline_config["min_budget"], max_budget=pipeline_config["max_budget"],
                                        host=ns_host, nameserver=ns_host, nameserver_port=ns_port,
                                        result_logger=result_logger,
                                        ping_interval=10**6,
                                        working_directory=pipeline_config["working_dir"],
                                        previous_result=previous_result,
                                        n_kdes=self.n_datasets,
                                        permutations=self.permutations)
        else:
            hb = optimization_algorithm(configspace=config_space, run_id = run_id,
                                        eta=pipeline_config["eta"], min_budget=pipeline_config["min_budget"], max_budget=pipeline_config["max_budget"],
                                        host=ns_host, nameserver=ns_host, nameserver_port=ns_port,
                                        result_logger=result_logger,
                                        ping_interval=10**6,
                                        working_directory=pipeline_config["working_dir"],
                                        previous_result=previous_result)
        return hb


    def parse_results(self, result_logger_dir):
        res = logged_results_to_HBS_result(result_logger_dir)
        id2config = res.get_id2config_mapping()
        incumbent_trajectory = res.get_incumbent_trajectory(bigger_is_better=False, non_decreasing_budget=False)
        
        if (len(incumbent_trajectory['config_ids']) == 0):
            return dict()
        
        final_config_id = incumbent_trajectory['config_ids'][-1]
        return incumbent_trajectory['losses'][-1], id2config[final_config_id]['config'], incumbent_trajectory['budgets'][-1]


    def run_worker(self, pipeline_config, constant_hyperparameter, run_id, task_id, ns_credentials_dir, network_interface_name,
            X_train, Y_train, X_valid, Y_valid):
        if not task_id == -1:
            time.sleep(5)
        while not os.path.isdir(ns_credentials_dir):
            time.sleep(5)
        host = nic_name_to_host(network_interface_name)
        
        worker = ModuleWorkerNoTimeLimit(   pipeline=self.sub_pipeline, pipeline_config=pipeline_config,
                                            constant_hyperparameter=constant_hyperparameter,
                                            X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, 
                                            budget_type=pipeline_config['budget_type'],
                                            max_budget=pipeline_config["max_budget"],
                                            host=host, run_id=run_id,
                                            id=task_id,
                                            working_directory=pipeline_config["result_logger_dir"],
                                            permutations=self.permutations)
        worker.load_nameserver_credentials(ns_credentials_dir)
        # run in background if not on cluster
        worker.run(background=(task_id <= 1))


    def run_optimization_algorithm(self, pipeline_config, config_space, constant_hyperparameter, run_id, ns_host, ns_port, nameserver, task_id):
        self.logger.info("[AutoNet] Start " + pipeline_config["algorithm"])

        # initialize optimization algorithm
        
        result_logger = self.get_result_logger(pipeline_config, constant_hyperparameter)
        HB = self.get_optimization_algorithm_instance(config_space=config_space, run_id=run_id,
            pipeline_config=pipeline_config, ns_host=ns_host, ns_port=ns_port, result_logger=result_logger)

        # start algorithm
        min_num_workers = pipeline_config["min_workers"] if task_id != -1 else 1

        reduce_runtime = pipeline_config["max_budget"] if pipeline_config["budget_type"] == "time" else 0
        
        HB.wait_for_workers(min_num_workers)
        self.logger.debug('Workers are ready!')

        thread_read_write.append('runs.log', "{0}: {1} | {2}-{3}\n".format(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            run_id,
            pipeline_config['min_budget'],
            pipeline_config['max_budget']))

        HB.run_until(runtime=(pipeline_config["max_runtime"] - reduce_runtime),
                     n_iterations=pipeline_config["num_iterations"],
                     min_n_workers=min_num_workers)

        HB.shutdown(shutdown_workers=True)
        nameserver.shutdown()

    
    def clean_fit_data(self):
        super(OptimizationAlgorithmNoTimeLimit, self).clean_fit_data()
        self.sub_pipeline.root.clean_fit_data()

    def run_refit(self, pipeline_config, refit, constants, X_train, Y_train, X_valid, Y_valid):
        start_time = time.time()

        result_logger = self.get_result_logger(pipeline_config, constants)
        result_logger.new_config((0, 0, 0), refit["hyperparameter_config"], {'model_based_pick': False})

        full_config = dict()
        full_config.update(constants)
        full_config.update(refit["hyperparameter_config"])

        self.logger.debug('Refit-Config:\n' + str(pprint.pformat(full_config)))

        class Job():
            pass
        job = Job()
        job.id = (0, 0, 0)
        job.kwargs = {
            'budget': refit['budget'],
            'config': refit["hyperparameter_config"],
        }

        try:
            res = self.sub_pipeline.fit_pipeline( 
                                    hyperparameter_config=full_config, pipeline_config=pipeline_config, 
                                    X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, 
                                    budget=refit["budget"], budget_type=pipeline_config['budget_type'], config_id='refit', working_directory=pipeline_config['result_logger_dir'])
            job.exception = None
        except Exception as e:
            self.logger.exception('Exception during refit')
            res = None
            job.exception = str(e)

        end_time = time.time()
        
        job.timestamps = {'submitted': start_time, 'started': start_time, 'finished': end_time}
        job.result = res

        result_logger(job)

        return {'loss': res['loss'] if res else float('inf'),
                'optimized_hyperparameter_config': full_config,
                'budget': refit['budget'],
                'info': res['info'] if res else dict()}

    def get_result_logger(self, pipeline_config, constant_hyperparameter):
        loggers = [bohb_logger(constant_hyperparameter=constant_hyperparameter, directory=pipeline_config["result_logger_dir"], overwrite=True)]
        if pipeline_config['use_tensorboard_logger']:
            loggers.append(tensorboard_logger(pipeline_config, constant_hyperparameter, pipeline_config['global_results_dir']))
        return combined_logger(*loggers)

    def get_permutations(self, n_budgets=1):
        # Get permutations, since HB fits like this: b1 - b2 -b3 - b2 -b3, repeat them accordingly
        idx = [i for i in range(self.n_datasets)]
        permutations = np.array(list(itertools.permutations(idx)))
        ret = []
        for perm in permutations:
            for ind in range(n_budgets):
                ret.append(perm)
        return np.array(ret)
