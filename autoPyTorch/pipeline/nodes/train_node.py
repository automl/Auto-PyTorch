__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import torch
import time
import logging

import scipy.sparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.components.training.base_training import BaseTrainingTechnique, BaseBatchLossComputationTechnique
from autoPyTorch.components.training.trainer import Trainer

import signal

class TrainNode(PipelineNode):
    """Training pipeline node. In this node, the network will be trained."""

    def __init__(self):
        """Construct the node"""
        super(TrainNode, self).__init__()
        self.default_minimize_value = True
        self.training_techniques = dict()
        self.batch_loss_computation_techniques = dict()
        self.add_batch_loss_computation_technique("standard", BaseBatchLossComputationTechnique)

    def fit(self, hyperparameter_config, pipeline_config,
            train_loader, valid_loader,
            network, optimizer,
            optimize_metric, additional_metrics,
            log_functions,
            budget,
            loss_function,
            training_techniques,
            fit_start_time,
            refit):
        """Train the network.
        
        Arguments:
            hyperparameter_config {dict} -- The sampled hyperparameter config.
            pipeline_config {dict} -- The user specified configuration of the pipeline
            train_loader {DataLoader} -- Data for training.
            valid_loader {DataLoader} -- Data for validation.
            network {BaseNet} -- The neural network to be trained.
            optimizer {AutoNetOptimizerBase} -- The selected optimizer.
            optimize_metric {AutoNetMetric} -- The selected metric to optimize
            additional_metrics {list} -- List of metrics, that should be logged
            log_functions {list} -- List of AutoNetLofFunctions that can log additional stuff like test performance
            budget {float} -- The budget for training
            loss_function {_Loss} -- The selected PyTorch loss module
            training_techniques {list} -- List of objects inheriting from BaseTrainingTechnique.
            fit_start_time {float} -- Start time of fit
            refit {bool} -- Whether training for refit or not.
        
        Returns:
            dict -- loss and info reported to bohb
        """
        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config) 
        logger = logging.getLogger('autonet')
        logger.debug("Start train. Budget: " + str(budget))

        if pipeline_config["torch_num_threads"] > 0:
            torch.set_num_threads(pipeline_config["torch_num_threads"])

        trainer = Trainer(
            model=network,
            loss_computation=self.batch_loss_computation_techniques[hyperparameter_config["batch_loss_computation_technique"]](),
            metrics=[optimize_metric] + additional_metrics,
            log_functions=log_functions,
            criterion=loss_function,
            budget=budget,
            optimizer=optimizer,
            training_techniques=training_techniques,
            device=Trainer.get_device(pipeline_config),
            logger=logger,
            full_eval_each_epoch=pipeline_config["full_eval_each_epoch"],
            log_every_n_points=pipeline_config["log_every_n_datapoints"],
            val_loader=valid_loader,
            logdir=pipeline_config["result_logger_dir"])
        trainer.prepare(pipeline_config, hyperparameter_config, fit_start_time)

        if pipeline_config["log_every_n_datapoints"] is None:
            model_params = self.count_parameters(network)
        else:
            model_params = 1

        logs = trainer.model.logs
        epoch = trainer.model.epochs_trained
        training_start_time = time.time()
        while True:
            # prepare epoch
            log = dict()
            trainer.on_epoch_start(log=log, epoch=epoch)
            
            # training
            optimize_metric_results, train_loss, stop_training = trainer.train(epoch + 1, train_loader)
            
            if pipeline_config["log_every_n_datapoints"] is None:
                if valid_loader is not None and trainer.eval_valid_each_epoch:
                    valid_metric_results = trainer.evaluate(valid_loader)

            # evaluate
            log['loss'] = train_loss
            log['model_parameters'] = model_params

            if pipeline_config["log_every_n_datapoints"] is None:
                for i, metric in enumerate(trainer.metrics):
                    log['train_' + metric.name] = optimize_metric_results[i]

                    if valid_loader is not None and trainer.eval_valid_each_epoch:
                        log['val_' + metric.name] = valid_metric_results[i]
                if trainer.eval_additional_logs_each_epoch:
                    for additional_log in trainer.log_functions:
                        log[additional_log.name] = additional_log(trainer.model, epoch)

            # wrap up epoch
            stop_training = trainer.on_epoch_end(log=log, epoch=epoch) or stop_training

            # handle logs
            logs.append(log)
            #log = {key: value for key, value in log.items() if not isinstance(value, np.ndarray)}
            logger.debug("Epoch: " + str(epoch) + " : " + str(log))

            if pipeline_config["log_every_n_datapoints"] is None:
                if 'use_tensorboard_logger' in pipeline_config and pipeline_config['use_tensorboard_logger']:
                    self.tensorboard_log(budget=budget, epoch=epoch, log=log, logdir=pipeline_config["result_logger_dir"])

            if stop_training:
                break
            
            epoch += 1
            torch.cuda.empty_cache()


        if pipeline_config["log_every_n_datapoints"] is not None:
            total_n_points = 1

            if valid_loader is not None and trainer.eval_valid_each_epoch:
                valid_metric_results = trainer.evaluate(valid_loader)

            for i, metric in enumerate(trainer.metrics):
                log['train_' + metric.name] = optimize_metric_results[i]
                if valid_loader is not None and trainer.eval_valid_each_epoch:
                    log['val_' + metric.name] = valid_metric_results[i]
            
            if trainer.eval_additional_logs_each_epoch:
                for additional_log in trainer.log_functions:
                    log[additional_log.name] = additional_log(trainer.model, epoch)

            logs.append(log)



        # wrap up
        loss, final_log = self.wrap_up_training(trainer=trainer, logs=logs, epoch=epoch,
            train_loader=train_loader, valid_loader=valid_loader, budget=budget, training_start_time=training_start_time, fit_start_time=fit_start_time,
            best_over_epochs=pipeline_config['best_over_epochs'], refit=refit, logger=logger)
    
        return {'loss': loss, 'info': final_log}


    def predict(self, pipeline_config, network, predict_loader):
        """Predict using trained neural network
        
        Arguments:
            pipeline_config {dict} -- The user specified configuration of the pipeline
            network {BaseNet} -- The trained neural network.
            predict_loader {DataLoader} -- The data to predict the labels for.
        
        Returns:
            dict -- The predicted labels in a dict.
        """
        if pipeline_config["torch_num_threads"] > 0:
            torch.set_num_threads(pipeline_config["torch_num_threads"])

        device = Trainer.get_device(pipeline_config)
        
        Y = predict(network, predict_loader, device)
        return {'Y': Y.detach().cpu().numpy()}
    
    def add_training_technique(self, name, training_technique):
        if (not issubclass(training_technique, BaseTrainingTechnique)):
            raise ValueError("training_technique type has to inherit from BaseTrainingTechnique")
        self.training_techniques[name] = training_technique
    
    def remove_training_technique(self, name):
        del self.training_techniques[name]
    
    def add_batch_loss_computation_technique(self, name, batch_loss_computation_technique):
        if (not issubclass(batch_loss_computation_technique, BaseBatchLossComputationTechnique)):
            raise ValueError("batch_loss_computation_technique type has to inherit from BaseBatchLossComputationTechnique, got " + str(batch_loss_computation_technique))
        self.batch_loss_computation_techniques[name] = batch_loss_computation_technique
    
    def remove_batch_loss_computation_technique(self, name, batch_loss_computation_technique):
        del self.batch_loss_computation_techniques[name]

    def get_hyperparameter_search_space(self, dataset_info=None, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        possible_techniques = set(pipeline_config['batch_loss_computation_techniques']).intersection(self.batch_loss_computation_techniques.keys())
        hp_batch_loss_computation = CSH.CategoricalHyperparameter("batch_loss_computation_technique", possible_techniques)
        cs.add_hyperparameter(hp_batch_loss_computation)

        for name, technique in self.batch_loss_computation_techniques.items():
            if name not in possible_techniques:
                continue
            technique = self.batch_loss_computation_techniques[name]

            technique_cs = technique.get_hyperparameter_search_space(
                **self._get_search_space_updates(prefix=("batch_loss_computation_technique", name)))
            cs.add_configuration_space(prefix=name, configuration_space=technique_cs,
                delimiter=ConfigWrapper.delimiter, parent_hyperparameter={'parent': hp_batch_loss_computation, 'value': name})

        self._check_search_space_updates((possible_techniques, "*"))
        return cs

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="batch_loss_computation_techniques", default=list(self.batch_loss_computation_techniques.keys()),
                type=str, list=True, choices=list(self.batch_loss_computation_techniques.keys())),
            ConfigOption("cuda", default=True, type=to_bool, choices=[True, False]),
            ConfigOption("torch_num_threads", default=1, type=int),
            ConfigOption("full_eval_each_epoch", default=False, type=to_bool, choices=[True, False],
                info="Whether to evaluate everything every epoch. Results in more useful output"),
            ConfigOption("best_over_epochs", default=False, type=to_bool, choices=[True, False],
                info="Whether to report the best performance occurred to BOHB"),
            #MODIFIED
            ConfigOption("log_every_n_datapoints", default=None, type=int,
                info="Log every n datapoints")
        ]
        for name, technique in self.training_techniques.items():
            options += technique.get_pipeline_config_options()
        return options
    
    def tensorboard_log(self, budget, epoch, log, logdir):
        import tensorboard_logger as tl
        worker_path = 'Train/'
        try:
            tl.log_value(worker_path + 'budget', float(budget), int(time.time()))
        except:
            tl.configure(logdir)
            tl.log_value(worker_path + 'budget', float(budget), int(time.time()))
        tl.log_value(worker_path + 'epoch', float(epoch + 1), int(time.time()))
        for name, value in log.items():
            if isinstance(value, (list, np.ndarray)):
                for ind, val in enumerate(value):
                    tl.log_value(worker_path + name + "_layer_" + str(ind), float(val), int(epoch+1))
            else:
                tl.log_value(worker_path + name, float(value), int(epoch+1))

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def wrap_up_training(self, trainer, logs, epoch, train_loader, valid_loader, budget,
            training_start_time, fit_start_time, best_over_epochs, refit, logger):
        """Wrap up and evaluate the training by computing missing log values
        
        Arguments:
            trainer {Trainer} -- The trainer used for training.
            logs {dict} -- The logs of the training
            epoch {int} -- Number of Epochs trained
            train_loader {DataLoader} -- The data for training
            valid_loader {DataLoader} -- The data for validation
            budget {float} -- Budget of training
            training_start_time {float} -- Start time of training
            fit_start_time {float} -- Start time of fit
            best_over_epochs {bool} -- Whether best validation data over epochs should be used
            refit {bool} -- Whether training was for refitting
            logger {Logger} -- Logger for logging stuff to the console
        
        Returns:
            tuple -- loss and selected final loss
        """
        wrap_up_start_time = time.time()
        trainer.model.epochs_trained = epoch
        trainer.model.logs = logs
        optimize_metric = trainer.metrics[0]
        opt_metric_name = 'train_' + optimize_metric.name
        if valid_loader is not None:
            opt_metric_name = 'val_' + optimize_metric.name

        final_log = trainer.final_eval(opt_metric_name=opt_metric_name,
            logs=logs, train_loader=train_loader, valid_loader=valid_loader, best_over_epochs=best_over_epochs, refit=refit)
        loss = trainer.metrics[0].loss_transform(final_log[opt_metric_name])

        logger.info("Finished train with budget " + str(budget) +
                         ": Preprocessing took " + str(int(training_start_time - fit_start_time)) +
                         "s, Training took " + str(int(wrap_up_start_time - training_start_time)) + 
                         "s, Wrap up took " + str(int(time.time() - wrap_up_start_time)) +
                         "s. Total time consumption in s: " + str(int(time.time() - fit_start_time)))
        return loss, final_log


def predict(network, test_loader, device, move_network=True):
    """ predict batchwise """
    # Build DataLoader
    if move_network:
        network = network.to(device)

    # Batch prediction
    network.eval()
    Y_batch_preds = list()
    
    for i, (X_batch, Y_batch) in enumerate(test_loader):
        # Predict on batch
        X_batch = Variable(X_batch).to(device)
        batch_size = X_batch.size(0)

        Y_batch_pred = network(X_batch).detach().cpu()
        Y_batch_preds.append(Y_batch_pred)
    
    return torch.cat(Y_batch_preds, 0)
