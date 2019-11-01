
import numpy as np
import os
import time
import shutil
import netifaces
import traceback
import logging

from hpbandster.core.nameserver import NameServer, nic_name_to_host
from hpbandster.core.result import logged_results_to_HBS_result

from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.pipeline.nodes import MetricSelector
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_condition import ConfigCondition

from autoPyTorch.core.hpbandster_extensions.bohb_ext import BOHBExt
from autoPyTorch.core.hpbandster_extensions.hyperband_ext import HyperBandExt
from autoPyTorch.core.worker import AutoNetWorker

from autoPyTorch.components.training.budget_types import BudgetTypeTime, BudgetTypeEpochs, BudgetTypeTrainingTime
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

        self.algorithms = {"bohb": BOHBExt,
                           "hyperband": HyperBandExt}

        self.budget_types = dict()
        self.budget_types["time"] = BudgetTypeTime
        self.budget_types["epochs"] = BudgetTypeEpochs
        self.budget_types["training_time"] = BudgetTypeTrainingTime

    def fit(self, pipeline_config, X_train, Y_train, X_valid, Y_valid, result_loggers, dataset_info, shutdownables, refit=None):
        """Run the optimization algorithm.
        
        Arguments:
            pipeline_config {dict} -- The configuration of the pipeline.
            X_train {array} -- The data
            Y_train {array} -- The data
            X_valid {array} -- The data
            Y_valid {array} -- The data
            result_loggers {list} -- List of loggers that log the result
            dataset_info {DatasetInfo} -- Object with information about the dataset
            shutdownables {list} -- List of objects that need to shutdown when optimization is finished.
        
        Keyword Arguments:
            refit {dict} -- dict containing information for refitting. None if optimization run should be started. (default: {None})
        
        Returns:
            dict -- Summary of optimization run.
        """
        logger = logging.getLogger('autonet')
        res = None

        run_id, task_id = pipeline_config['run_id'], pipeline_config['task_id']

        # Use tensorboard logger
        if pipeline_config['use_tensorboard_logger'] and not refit:            
            import tensorboard_logger as tl
            directory = os.path.join(pipeline_config['result_logger_dir'], "worker_logs_" + str(task_id))
            os.makedirs(directory, exist_ok=True)
            tl.configure(directory, flush_secs=5)

        # Only do refitting
        if (refit is not None):
            logger.info("Start Refitting")

            loss_info_dict = self.sub_pipeline.fit_pipeline( 
                                    hyperparameter_config=refit["hyperparameter_config"], pipeline_config=pipeline_config, 
                                    X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, 
                                    budget=refit["budget"], rescore=refit["rescore"], budget_type=self.budget_types[pipeline_config['budget_type']],
                                    optimize_start_time=time.time(), refit=True, hyperparameter_config_id=None, dataset_info=dataset_info)
            logger.info("Done Refitting")
            
            return {'optimized_hyperparameter_config': refit["hyperparameter_config"],
                    'budget': refit['budget'],
                    'loss': loss_info_dict['loss'],
                    'info': loss_info_dict['info']}

        # Start Optimization Algorithm
        try:
            ns_credentials_dir, tmp_models_dir, network_interface_name = self.prepare_environment(pipeline_config)

            # start nameserver if not on cluster or on master node in cluster
            if task_id in [1, -1]:
                NS = self.get_nameserver(run_id, task_id, ns_credentials_dir, network_interface_name)
                ns_host, ns_port = NS.start()
                
            if task_id != 1 or pipeline_config["run_worker_on_master_node"]:
                self.run_worker(pipeline_config=pipeline_config, run_id=run_id, task_id=task_id, ns_credentials_dir=ns_credentials_dir,
                    network_interface_name=network_interface_name, X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid,
                    dataset_info=dataset_info, shutdownables=shutdownables)

            # start BOHB if not on cluster or on master node in cluster
            res = None
            if task_id in [1, -1]:
                self.run_optimization_algorithm(pipeline_config=pipeline_config, run_id=run_id, ns_host=ns_host,
                    ns_port=ns_port, nameserver=NS, task_id=task_id, result_loggers=result_loggers,
                    dataset_info=dataset_info, logger=logger)
   
            
                res = self.parse_results(pipeline_config)

        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            self.clean_up(pipeline_config, ns_credentials_dir, tmp_models_dir)

        if res:
            return res
        return {'optimized_hyperparameter_config': dict(), 'budget': 0, 'loss': float('inf'), 'info': dict()}

    def predict(self, pipeline_config, X):
        """Run the predict pipeline.
        
        Arguments:
            pipeline_config {dict} -- The configuration of the pipeline
            X {array} -- The data
        
        Returns:
            dict -- The predicted values in a dictionary
        """
        result = self.sub_pipeline.predict_pipeline(pipeline_config=pipeline_config, X=X)
        return {'Y': result['Y']}

    # OVERRIDE
    def get_pipeline_config_options(self):
        options = [
            ConfigOption("run_id", default="0", type=str, info="Unique id for each run."),
            ConfigOption("task_id", default=-1, type=int, info="ID for each worker, if you run AutoNet on a cluster. Set to -1, if you run it locally. "),
            ConfigOption("algorithm", default="bohb", type=str, choices=list(self.algorithms.keys()), info="Algorithm to use for config sampling."),
            ConfigOption("budget_type", default="time", type=str, choices=list(self.budget_types.keys())),
            ConfigOption("min_budget", default=lambda c: self.budget_types[c["budget_type"]].default_min_budget, type=float, depends=True, info="Min budget for fitting configurations."),
            ConfigOption("max_budget", default=lambda c: self.budget_types[c["budget_type"]].default_max_budget, type=float, depends=True, info="Max budget for fitting configurations."),
            ConfigOption("max_runtime", 
                default=lambda c: ((-int(np.log(c["min_budget"] / c["max_budget"]) / np.log(c["eta"])) + 1) * c["max_budget"])
                        if c["budget_type"] == "time" else float("inf"),
                type=float, depends=True, info="Total time for the run."),
            ConfigOption("num_iterations", 
                default=lambda c:  (-int(np.log(c["min_budget"] / c["max_budget"]) / np.log(c["eta"])) + 1)
                        if c["budget_type"] == "epochs" else float("inf"),
                type=float, depends=True, info="Number of successive halving iterations."),
            ConfigOption("eta", default=3, type=float, info='eta parameter of Hyperband.'),
            ConfigOption("min_workers", default=1, type=int),
            ConfigOption("working_dir", default=".", type="directory"),
            ConfigOption("network_interface_name", default=self.get_default_network_interface_name(), type=str),
            ConfigOption("memory_limit_mb", default=1000000, type=int),
            ConfigOption("use_tensorboard_logger", default=False, type=to_bool),
            ConfigOption("run_worker_on_master_node", default=True, type=to_bool),
            ConfigOption("use_pynisher", default=True, type=to_bool)
        ]
        return options

    # OVERRIDE
    def get_pipeline_config_conditions(self):
        def check_runtime(pipeline_config):
            return pipeline_config["budget_type"] != "time" or pipeline_config["max_runtime"] >= pipeline_config["max_budget"]

        return [
            ConfigCondition.get_larger_equals_condition("max budget must be greater than or equal to min budget", "max_budget", "min_budget"),
            ConfigCondition("When time is used as budget, the max_runtime must be larger than the max_budget", check_runtime)
        ]


    def get_default_network_interface_name(self):
        """Get the default network interface name
        
        Returns:
            str -- The default network interface name
        """
        try:
            return netifaces.gateways()['default'][netifaces.AF_INET][1]
        except:
            return 'lo'

    def prepare_environment(self, pipeline_config):
        """Create necessary folders and get network interface name
        
        Arguments:
            pipeline_config {dict} -- The configuration of the pipeline
        
        Returns:
            tuple -- path to created directories and network interface namei
        """
        if not os.path.exists(pipeline_config["working_dir"]) and pipeline_config['task_id'] in [1, -1]:
            try:
                os.mkdir(pipeline_config["working_dir"])
            except:
                pass
        tmp_models_dir = os.path.join(pipeline_config["working_dir"], "tmp_models_" + str(pipeline_config['run_id']))
        ns_credentials_dir = os.path.abspath(os.path.join(pipeline_config["working_dir"], "ns_credentials_" + str(pipeline_config['run_id'])))
        network_interface_name = self.get_nic_name(pipeline_config)
        
        if os.path.exists(tmp_models_dir) and pipeline_config['task_id'] in [1, -1]:
            shutil.rmtree(tmp_models_dir)  # not used right now
        if os.path.exists(ns_credentials_dir) and pipeline_config['task_id'] in [1, -1]:
            shutil.rmtree(ns_credentials_dir)
        return ns_credentials_dir, tmp_models_dir, network_interface_name

    def clean_up(self, pipeline_config, tmp_models_dir, ns_credentials_dir):
        """Remove created folders
        
        Arguments:
            pipeline_config {dict} -- The pipeline config
            tmp_models_dir {[type]} -- The path to the temporary models (not used right now)
            ns_credentials_dir {[type]} --  The path to the nameserver credentials
        """
        if pipeline_config['task_id'] in [1, -1]:
            # Delete temporary files
            if os.path.exists(tmp_models_dir):
                shutil.rmtree(tmp_models_dir)
            if os.path.exists(ns_credentials_dir):
                shutil.rmtree(ns_credentials_dir)

    def get_nameserver(self, run_id, task_id, ns_credentials_dir, network_interface_name):
        """Get the namesever object
        
        Arguments:
            run_id {str} -- The id of the run
            task_id {int} -- An id for the worker
            ns_credentials_dir {str} -- Path to ns credentials
            network_interface_name {str} -- The network interface name
        
        Returns:
            NameServer -- The NameServer object
        """
        if not os.path.isdir(ns_credentials_dir):
            try:
                os.mkdir(ns_credentials_dir)
            except:
                pass
        return NameServer(run_id=run_id, nic_name=network_interface_name, working_directory=ns_credentials_dir)
    
    def get_optimization_algorithm_instance(self, config_space, run_id, pipeline_config, ns_host, ns_port, loggers, previous_result=None):
        """Get an instance of the optimization algorithm
        
        Arguments:
            config_space {ConfigurationSpace} -- The config space to optimize.
            run_id {str} -- An Id for the current run.
            pipeline_config {dict} -- The configuration of the pipeline.
            ns_host {str} -- Nameserver host.
            ns_port {int} -- Nameserver port.
            loggers {list} -- Loggers to log the results.
        
        Keyword Arguments:
            previous_result {Result} -- A previous result to warmstart the search (default: {None})
        
        Returns:
            Master -- An optimization algorithm.
        """
        optimization_algorithm = self.algorithms[pipeline_config["algorithm"]]
        kwargs = {"configspace": config_space, "run_id": run_id,
                  "eta": pipeline_config["eta"], "min_budget": pipeline_config["min_budget"], "max_budget": pipeline_config["max_budget"],
                  "host": ns_host, "nameserver": ns_host, "nameserver_port": ns_port,
                  "result_logger": combined_logger(*loggers),
                  "ping_interval": 10**6,
                  "working_directory": pipeline_config["working_dir"],
                  "previous_result": previous_result}
        hb = optimization_algorithm(**kwargs)
        return hb


    def parse_results(self, pipeline_config):
        """Parse the results of the optimization run
        
        Arguments:
            pipeline_config {dict} -- The configuration of the pipeline.
        
        Raises:
            RuntimeError:  An Error occurred when parsing the results.
        
        Returns:
            dict -- Dictionary summarizing the results
        """
        try:
            res = logged_results_to_HBS_result(pipeline_config["result_logger_dir"])
            id2config = res.get_id2config_mapping()
            incumbent_trajectory = res.get_incumbent_trajectory(bigger_is_better=False, non_decreasing_budget=False)
        except Exception as e:
            raise RuntimeError("Error parsing results. Check results.json and output for more details. An empty results.json is usually caused by a misconfiguration of AutoNet.")

        if (len(incumbent_trajectory['config_ids']) == 0):
            return dict()
        
        final_config_id = incumbent_trajectory['config_ids'][-1]
        final_budget = incumbent_trajectory['budgets'][-1]
        best_run = [r for r in res.get_runs_by_id(final_config_id) if r.budget == final_budget][0]
        return {'optimized_hyperparameter_config': id2config[final_config_id]['config'],
                'budget': final_budget,
                'loss': best_run.loss,
                'info': best_run.info}


    def run_worker(self, pipeline_config, run_id, task_id, ns_credentials_dir, network_interface_name,
            X_train, Y_train, X_valid, Y_valid, dataset_info, shutdownables):
        """ Run the AutoNetWorker
        
        Arguments:
            pipeline_config {dict} -- The configuration of the pipeline
            run_id {str} -- An id for the run
            task_id {int} -- An id for the worker
            ns_credentials_dir {str} -- path to nameserver credentials
            network_interface_name {str} -- the name of the network interface
            X_train {array} -- The data
            Y_train {array} -- The data
            X_valid {array} -- The data
            Y_valid {array} -- The data
            dataset_info {DatasetInfo} -- Object describing the dataset
            shutdownables {list} -- A list of objects that need to shutdown when the optimization is finished
        """
        if not task_id == -1:
            time.sleep(5)
        while not os.path.isdir(ns_credentials_dir):
            time.sleep(5)
        host = nic_name_to_host(network_interface_name)
        
        worker = AutoNetWorker(pipeline=self.sub_pipeline, pipeline_config=pipeline_config,
                              X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, dataset_info=dataset_info,
                              budget_type=self.budget_types[pipeline_config['budget_type']],
                              max_budget=pipeline_config["max_budget"],
                              host=host, run_id=run_id,
                              id=task_id, shutdownables=shutdownables,
                              use_pynisher=pipeline_config["use_pynisher"])
        worker.load_nameserver_credentials(ns_credentials_dir)
        # run in background if not on cluster
        worker.run(background=(task_id <= 1))


    def run_optimization_algorithm(self, pipeline_config, run_id, ns_host, ns_port, nameserver, task_id, result_loggers,
            dataset_info, logger):
        """ 
        
        Arguments:
            pipeline_config {dict} -- The configuration of the pipeline
            run_id {str} -- An id for the run
            ns_host {str} -- Nameserver host.
            ns_port {int} -- Nameserver port.
            nameserver {[type]} -- The nameserver.
            task_id {int} -- An id for the worker
            result_loggers {[type]} -- [description]
            dataset_info {DatasetInfo} -- Object describing the dataset
            logger {list} -- Loggers to log the results.
        """
        config_space = self.pipeline.get_hyperparameter_search_space(dataset_info=dataset_info, **pipeline_config)


        logger.info("[AutoNet] Start " + pipeline_config["algorithm"])

        # initialize optimization algorithm
        if pipeline_config['use_tensorboard_logger']:
            result_loggers.append(tensorboard_logger())

        HB = self.get_optimization_algorithm_instance(config_space=config_space, run_id=run_id,
            pipeline_config=pipeline_config, ns_host=ns_host, ns_port=ns_port, loggers=result_loggers)

        # start algorithm
        min_num_workers = pipeline_config["min_workers"] if task_id != -1 else 1

        reduce_runtime = pipeline_config["max_budget"] if pipeline_config["budget_type"] == "time" else 0
        HB.run_until(runtime=(pipeline_config["max_runtime"] - reduce_runtime),
                     n_iterations=pipeline_config["num_iterations"],
                     min_n_workers=min_num_workers)

        HB.shutdown(shutdown_workers=True)
        nameserver.shutdown()
    
    @staticmethod
    def get_nic_name(pipeline_config):
        """Get the nic name from the pipeline config"""
        return pipeline_config["network_interface_name"] or (netifaces.interfaces()[1] if len(netifaces.interfaces()) > 1 else "lo")

    
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


class combined_logger(object):
    def __init__(self, *loggers):
        self.loggers = loggers

    def new_config(self, config_id, config, config_info):
        for logger in self.loggers:
            logger.new_config(config_id, config, config_info)

    def __call__(self, job):
        for logger in self.loggers:
            logger(job)
        
