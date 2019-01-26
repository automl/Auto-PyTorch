
import numpy as np
import os
import time
import shutil
import netifaces
import traceback
import logging

from hpbandster.core.nameserver import NameServer, nic_name_to_host
from hpbandster.core.result import (json_result_logger,
                                    logged_results_to_HBS_result)

from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_condition import ConfigCondition

from autoPyTorch.core.hpbandster_extensions.bohb_ext import BOHBExt
from autoPyTorch.core.hpbandster_extensions.hyperband_ext import HyperBandExt
from autoPyTorch.core.worker import ModuleWorker

from autoPyTorch.training.budget_types import BudgetTypeTime, BudgetTypeEpochs
import copy

class OptimizationAlgorithm(SubPipelineNode):
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

        super(OptimizationAlgorithm, self).__init__(optimization_pipeline_nodes)

        self.algorithms = dict()
        self.algorithms["bohb"] = BOHBExt
        self.algorithms["hyperband"] = HyperBandExt

        self.budget_types = dict()
        self.budget_types["time"] = BudgetTypeTime
        self.budget_types["epochs"] = BudgetTypeEpochs
        
        self.logger = logging.getLogger('autonet')

    def fit(self, pipeline_config, X_train, Y_train, X_valid, Y_valid, refit=None):
        res = None

        run_id, task_id = pipeline_config['run_id'], pipeline_config['task_id']

        if pipeline_config['use_tensorboard_logger']:            
            import tensorboard_logger as tl
            directory = os.path.join(pipeline_config['result_logger_dir'], "worker_logs_" + str(task_id))
            os.makedirs(directory, exist_ok=True)
            tl.configure(directory, flush_secs=5)

        if (refit is not None):
            res = self.sub_pipeline.fit_pipeline( 
                                    hyperparameter_config=refit["hyperparameter_config"], pipeline_config=pipeline_config, 
                                    X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, 
                                    budget=refit["budget"], budget_type=self.budget_types[pipeline_config['budget_type']],
                                    optimize_start_time=time.time())
            
            return {'final_metric_score': res['loss'],
                    'optimized_hyperparamater_config': refit["hyperparameter_config"],
                    'budget': refit['budget']}

        try:
            ns_credentials_dir, tmp_models_dir, network_interface_name = self.prepare_environment(pipeline_config)

            # start nameserver if not on cluster or on master node in cluster
            if task_id in [1, -1]:
                NS = self.get_nameserver(run_id, task_id, ns_credentials_dir, network_interface_name)
                ns_host, ns_port = NS.start()
                

            self.run_worker(pipeline_config=pipeline_config, run_id=run_id, task_id=task_id, ns_credentials_dir=ns_credentials_dir,
                network_interface_name=network_interface_name, X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)

            # start BOHB if not on cluster or on master node in cluster
            if task_id in [1, -1]:
                self.run_optimization_algorithm(pipeline_config, run_id, ns_host, ns_port, NS, task_id)
            

            res = self.parse_results(pipeline_config["result_logger_dir"])

        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            self.clean_up(pipeline_config, ns_credentials_dir, tmp_models_dir)

        if (res):
            return {'final_metric_score': res[0], 'optimized_hyperparamater_config': res[1], 'budget': res[2]}
        else:
            return {'final_metric_score': None, 'optimized_hyperparamater_config': dict(), 'budget': 0}

    def predict(self, pipeline_config, X):
        result = self.sub_pipeline.predict_pipeline(pipeline_config=pipeline_config, X=X)
        return {'Y': result['Y']}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("run_id", default="0", type=str, info="Unique id for each run."),
            ConfigOption("task_id", default=-1, type=int, info="ID for each worker, if you run AutoNet on a cluster. Set to -1, if you run it locally. "),
            ConfigOption("algorithm", default="bohb", type=str, choices=list(self.algorithms.keys())),
            ConfigOption("budget_type", default="time", type=str, choices=list(self.budget_types.keys())),
            ConfigOption("min_budget", default=lambda c: self.budget_types[c["budget_type"]].default_min_budget, type=float, depends=True),
            ConfigOption("max_budget", default=lambda c: self.budget_types[c["budget_type"]].default_max_budget, type=float, depends=True),
            ConfigOption("result_logger_dir", default=".", type="directory"),
            ConfigOption("max_runtime", 
                default=lambda c: ((-int(np.log(c["min_budget"] / c["max_budget"]) / np.log(c["eta"])) + 1) * c["max_budget"])
                        if c["budget_type"] == "time" else float("inf"),
                type=float, depends=True),
            ConfigOption("num_iterations", 
                default=lambda c:  (-int(np.log(c["min_budget"] / c["max_budget"]) / np.log(c["eta"])) + 1)
                        if c["budget_type"] == "epochs" else float("inf"),
                type=float, depends=True),
            ConfigOption("eta", default=3, type=float, info='eta parameter of Hyperband.'),
            ConfigOption("min_workers", default=1, type=int),
            ConfigOption("working_dir", default=".", type="directory"),
            ConfigOption("network_interface_name", default=self.get_default_network_interface_name(), type=str),
            ConfigOption("memory_limit_mb", default=1000000, type=int),
            ConfigOption("use_tensorboard_logger", default=False, type=to_bool),
        ]
        return options
    
    def get_pipeline_config_conditions(self):
        return [
            ConfigCondition.get_larger_equals_condition("max budget must be greater than or equal to min budget", "max_budget", "min_budget")
        ]

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
    
    def get_optimization_algorithm_instance(self, config_space, run_id, pipeline_config, ns_host, ns_port, loggers, previous_result=None):
        optimization_algorithm = self.algorithms[pipeline_config["algorithm"]]
        hb = optimization_algorithm(configspace=config_space, run_id = run_id,
                                    eta=pipeline_config["eta"], min_budget=pipeline_config["min_budget"], max_budget=pipeline_config["max_budget"],
                                    host=ns_host, nameserver=ns_host, nameserver_port=ns_port,
                                    result_logger=combined_logger(*loggers),
                                    ping_interval=10**6,
                                    working_directory=pipeline_config["working_dir"],
                                    previous_result=previous_result)
        return hb


    def parse_results(self, result_logger_dir):
        try:
            res = logged_results_to_HBS_result(result_logger_dir)
            id2config = res.get_id2config_mapping()
            incumbent_trajectory = res.get_incumbent_trajectory(bigger_is_better=False, non_decreasing_budget=False)
        except Exception as e:
            raise RuntimeError("Error parsing results. Check results.json and output for more details. An empty results.json is usually caused by a misconfiguration of AutoNet.")
        
        if (len(incumbent_trajectory['config_ids']) == 0):
            return dict()
        
        final_config_id = incumbent_trajectory['config_ids'][-1]
        return incumbent_trajectory['losses'][-1], id2config[final_config_id]['config'], incumbent_trajectory['budgets'][-1]


    def run_worker(self, pipeline_config, run_id, task_id, ns_credentials_dir, network_interface_name,
            X_train, Y_train, X_valid, Y_valid):
        if not task_id == -1:
            time.sleep(5)
        while not os.path.isdir(ns_credentials_dir):
            time.sleep(5)
        host = nic_name_to_host(network_interface_name)
        
        worker = ModuleWorker(pipeline=self.sub_pipeline, pipeline_config=pipeline_config,
                              X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, 
                              budget_type=self.budget_types[pipeline_config['budget_type']],
                              max_budget=pipeline_config["max_budget"],
                              host=host, run_id=run_id,
                              id=task_id)
        worker.load_nameserver_credentials(ns_credentials_dir)
        # run in background if not on cluster
        worker.run(background=(task_id <= 1))


    def run_optimization_algorithm(self, pipeline_config, run_id, ns_host, ns_port, nameserver, task_id):
        config_space = self.pipeline.get_hyperparameter_search_space(**pipeline_config)


        self.logger.info("[AutoNet] Start " + pipeline_config["algorithm"])

        # initialize optimization algorithm
        loggers = [json_result_logger(directory=pipeline_config["result_logger_dir"], overwrite=True)]
        if pipeline_config['use_tensorboard_logger']:
            loggers.append(tensorboard_logger())

        HB = self.get_optimization_algorithm_instance(config_space=config_space, run_id=run_id,
            pipeline_config=pipeline_config, ns_host=ns_host, ns_port=ns_port, loggers=loggers)

        # start algorithm
        min_num_workers = pipeline_config["min_workers"] if task_id != -1 else 1

        reduce_runtime = pipeline_config["max_budget"] if pipeline_config["budget_type"] == "time" else 0
        HB.run_until(runtime=(pipeline_config["max_runtime"] - reduce_runtime),
                     n_iterations=pipeline_config["num_iterations"],
                     min_n_workers=min_num_workers)

        HB.shutdown(shutdown_workers=True)
        nameserver.shutdown()

    
    def clean_fit_data(self):
        super(OptimizationAlgorithm, self).clean_fit_data()
        self.sub_pipeline.root.clean_fit_data()




class tensorboard_logger(object):
    def __init__(self):
        self.start_time = time.time()
        self.incumbent = float('inf')

    def new_config(self, config_id, config, config_info):
        pass

    def __call__(self, job):
        import tensorboard_logger as tl 
        # id = job.id
        budget = job.kwargs['budget']
        # config = job.kwargs['config']
        timestamps = job.timestamps
        result = job.result
        exception = job.exception

        time_step = int(timestamps['finished'] - self.start_time)

        if result is not None:
            tl.log_value('BOHB/all_results', result['loss'] * -1, time_step)
            if result['loss'] < self.incumbent:
                self.incumbent = result['loss']
                tl.log_value('BOHB/incumbent_results', self.incumbent * -1, time_step)
        else:
            tl.log_value('Exceptions/' + str(exception.split('\n')[-2]), budget, time_step)


class combined_logger(object):
    def __init__(self, *loggers):
        self.loggers = loggers

    def new_config(self, config_id, config, config_info):
        for logger in self.loggers:
            logger.new_config(config_id, config, config_info)

    def __call__(self, job):
        for logger in self.loggers:
            logger(job)
        
