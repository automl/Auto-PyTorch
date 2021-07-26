
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
from autoPyTorch.training.base_training import BaseTrainingTechnique, BaseBatchLossComputationTechnique

import signal

class TrainNode(PipelineNode):
    def __init__(self):
        super(TrainNode, self).__init__()
        self.default_minimize_value = True
        self.logger = logging.getLogger('autonet')
        self.training_techniques = dict()
        self.batch_loss_computation_techniques = dict()
        self.add_batch_loss_computation_technique("standard", BaseBatchLossComputationTechnique)

    def fit(self, hyperparameter_config, pipeline_config,
            X_train, Y_train, X_valid, Y_valid,
            network, optimizer,
            train_metric, additional_metrics,
            log_functions,
            budget,
            loss_function,
            training_techniques,
            fit_start_time):
        # prepare
        if not torch.cuda.is_available():
            pipeline_config["cuda"] = False
        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)
        training_techniques = [t() for t in self.training_techniques.values()] + training_techniques
        training_components, train_data, X_train, Y_train, X_valid, Y_valid, eval_specifics = prepare_training(
            pipeline_config=pipeline_config, hyperparameter_config=hyperparameter_config, training_techniques=training_techniques,
            batch_loss_computation_technique=self.batch_loss_computation_techniques[hyperparameter_config["batch_loss_computation_technique"]](),
            X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, batch_size=hyperparameter_config["batch_size"],
            network=network, optimizer=optimizer, loss_function=loss_function, train_metric=train_metric,
            additional_metrics=additional_metrics, log_functions=log_functions, budget=budget, logger=self.logger, fit_start_time=fit_start_time)
        self.logger.debug("Start train. Budget: " + str(budget))

        # Training loop
        logs = network.logs
        epoch = network.epochs_trained
        run_training = True
        training_start_time = time.time()

        while run_training:

            # prepare epoch
            log = dict()
            for t in training_techniques:
                t.before_train_batches(training_components, log, epoch)

            # train and eval
            log['loss'] = _train_batches(train_data, training_components, training_techniques)
            _eval_metrics(eval_specifics=eval_specifics["after_epoch"], hyperparameter_config=hyperparameter_config,
                pipeline_config=pipeline_config, training_components=training_components,
                X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, log=log, epoch=epoch, budget=budget)
            
            # check if finished and apply training techniques
            run_training = not any([t.after_train_batches(training_components, log, epoch) for t in training_techniques])

            # handle logs
            logs.append(log)
            # update_logs(t, budget, log, 5, epoch + 1, verbose, True)
            self.logger.debug("Epoch: " + str(epoch) + " : " + str(log))
            epoch += 1

        # wrap up
        wrap_up_start_time = time.time()
        network.epochs_trained = epoch
        network.logs = logs
        final_log, loss_value = wrap_up_training(pipeline_config=pipeline_config, hyperparameter_config=hyperparameter_config,
            eval_specifics=eval_specifics["after_training"], training_techniques=training_techniques, training_components=training_components,
            logs=logs, X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, epoch=epoch, budget=budget)
        self.logger.debug("Finished train! Loss: " + str(loss_value) + " : " + str(final_log))
        self.logger.info("Finished train with budget " + str(budget) +
                         ": Preprocessing took " + str(int(training_start_time - fit_start_time)) +
                         "s, Training took " + str(int(wrap_up_start_time - training_start_time)) + 
                         "s, Wrap up took " + str(int(time.time() - wrap_up_start_time)) +
                         "s. Total time consumption in s: " + str(int(time.time() - fit_start_time)))
        return {'loss': loss_value, 'info': final_log}


    def predict(self, pipeline_config, network, X):
        device = torch.device('cuda:0' if pipeline_config['cuda'] else 'cpu')
        
        X = torch.from_numpy(X).float()
        Y = predict(network, X, 20, device)
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

    def get_hyperparameter_search_space(self, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter("batch_size", lower=32, upper=500, log=True))
        hp_batch_loss_computation = CSH.CategoricalHyperparameter("batch_loss_computation_technique",
            pipeline_config['batch_loss_computation_techniques'], default_value=pipeline_config['batch_loss_computation_techniques'][0])
        cs.add_hyperparameter(hp_batch_loss_computation)

        for name in pipeline_config['batch_loss_computation_techniques']:
            technique = self.batch_loss_computation_techniques[name]
            cs.add_configuration_space(prefix=name, configuration_space=technique.get_hyperparameter_search_space(**pipeline_config),
                delimiter=ConfigWrapper.delimiter, parent_hyperparameter={'parent': hp_batch_loss_computation, 'value': name})

        return self._apply_user_updates(cs)

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="batch_loss_computation_techniques", default=list(self.batch_loss_computation_techniques.keys()),
                type=str, list=True, choices=list(self.batch_loss_computation_techniques.keys())),
            ConfigOption(name="training_techniques", default=list(self.training_techniques.keys()),
                type=str, list=True, choices=list(self.training_techniques.keys())),
            ConfigOption("minimize", default=self.default_minimize_value, type=to_bool, choices=[True, False],
                info="Whether the specified train metric should be minimized."),
            ConfigOption("cuda", default=True, type=to_bool, choices=[True, False]),
            ConfigOption("eval_on_training", default=False, type=to_bool, choices=[True, False],
                info="Whether to evaluate on training data. Results in more printed info in certain loglevels."),
            ConfigOption("full_eval_each_epoch", default=False, type=to_bool, choices=[True, False],
                info="Whether full evaluation should be performed after each epoch. Results in more printed info in certain loglevels.")
        ]
        for name, technique in self.training_techniques.items():
            options += technique.get_pipeline_config_options()
        for name, technique in self.batch_loss_computation_techniques.items():
            options += technique.get_pipeline_config_options()
        return options


def prepare_training(pipeline_config, hyperparameter_config, training_techniques, batch_loss_computation_technique,
        X_train, Y_train, X_valid, Y_valid, batch_size, network, optimizer, loss_function, train_metric, additional_metrics,
        log_functions, budget, logger, fit_start_time):
    """ Prepare the data and components for training"""

    torch.manual_seed(pipeline_config["random_seed"]) 
    device = torch.device('cuda:0' if pipeline_config['cuda'] else 'cpu')

    if pipeline_config['cuda']:
        logger.debug('Running on the GPU using CUDA.')
    else:
        logger.debug('Not running on GPU as CUDA is either disabled or not available. Running on CPU instead.')

    # initialize training techniques and training components
    batch_loss_computation_technique.set_up(
        pipeline_config, ConfigWrapper(hyperparameter_config["batch_loss_computation_technique"], hyperparameter_config), logger)
    training_components = {
        "network": network.to(device),
        "optimizer": optimizer,
        "loss_function": loss_function.to(device),
        "metrics": [train_metric] + additional_metrics,
        "train_metric_name": train_metric.__name__,
        "log_functions": log_functions,
        "device": device,
        "initial_budget": network.budget_trained,
        "budget": budget,
        "batch_loss_computation_technique": batch_loss_computation_technique,
        "fit_start_time": fit_start_time
    }
    [training_components.update(t.training_components) for t in training_techniques]
    for t in training_techniques:
        t.set_up(training_components, pipeline_config, logger)

    # prepare data
    X_train, Y_train, X_valid, Y_valid = to_dense(X_train), to_dense(Y_train), to_dense(X_valid), to_dense(Y_valid)
    X_train, Y_train = torch.from_numpy(X_train).float(), torch.from_numpy(Y_train)
    train_data = DataLoader(TensorDataset(X_train, Y_train), batch_size, True)
    X_valid = torch.from_numpy(X_valid).float().to(device) if X_valid is not None else None
    Y_valid = torch.from_numpy(Y_valid).to(device) if Y_valid is not None else None

    # eval specifics. decide which datasets should be evaluated when.
    after_epoch_eval_specifics = {
        "train": any(t.needs_eval_on_train_each_epoch() for t in training_techniques)
            or (pipeline_config["full_eval_each_epoch"] and pipeline_config["eval_on_training"]),
        "valid": any(t.needs_eval_on_valid_each_epoch() for t in training_techniques) or pipeline_config["full_eval_each_epoch"],
        "logs": pipeline_config["full_eval_each_epoch"]
    }
    after_training_eval_specifics = {
        "train": not after_epoch_eval_specifics["train"] and (pipeline_config["eval_on_training"] or X_valid is None or Y_valid is None),
        "valid": not after_epoch_eval_specifics["valid"],
        "logs": not after_epoch_eval_specifics["logs"]
    }
    eval_specifics = {"after_epoch": after_epoch_eval_specifics, "after_training": after_training_eval_specifics}
    return training_components, train_data, X_train, Y_train, X_valid, Y_valid, eval_specifics

def to_dense(matrix):
    if (matrix is not None and scipy.sparse.issparse(matrix)):
        return matrix.todense()
    return matrix

def wrap_up_training(pipeline_config, hyperparameter_config, eval_specifics, training_techniques, training_components, logs,
        X_train, Y_train, X_valid, Y_valid, epoch, budget):
    """ Finalize the logs returned to bohb """

    # select which log to use.
    final_log = None
    for t in training_techniques:
        log = t.select_log(logs, training_components)
        if log:
            final_log = log
    if not final_log:
        final_log = logs[-1]

    # evaluate on datasets, that should be evaluated after training.
    if any([eval_specifics["train"], eval_specifics["valid"], eval_specifics["logs"]]):
        training_components["network"].load_snapshot()
        _eval_metrics(eval_specifics=eval_specifics, hyperparameter_config=hyperparameter_config, pipeline_config=pipeline_config,
            training_components=training_components, X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, log=final_log, epoch=epoch, budget=budget)

    # get the loss value out of the selected log
    if "val_" + training_components["train_metric_name"] in final_log:
        loss_value = final_log["val_" + training_components["train_metric_name"]] * (1 if pipeline_config["minimize"] else -1)
    else:
        loss_value = final_log["train_" + training_components["train_metric_name"]] * (1 if pipeline_config["minimize"] else -1)
    return final_log, loss_value

def _train_batches(train_data, training_components, training_techniques):
    """ perform training in batches """
    training_components["network"].train()
    epoch_loss = 0.0
    # Run batches
    for batch_i, batch_data in enumerate(train_data):
        if batch_data[0].size()[0] == 1:
            continue
        X_batch = batch_data[0].to(training_components["device"])
        y_batch = batch_data[1].to(training_components["device"])
        training_components["batch_loss_computation_technique"].prepare_batch_data(X_batch, y_batch)
            
        # Backprop
        training_components["optimizer"].zero_grad()
        y_batch_pred = training_components["network"](Variable(X_batch))

        batch_loss = training_components["batch_loss_computation_technique"].compute_batch_loss(training_components["loss_function"], y_batch_pred)
        batch_loss.backward()
        training_components["optimizer"].step()

        # Update status
        epoch_loss += batch_loss.data.item()

        if any([t.during_train_batches(batch_loss, training_components) for t in training_techniques]):
            break

    return float(epoch_loss) / (batch_i + 1)


def predict(network, X, batch_size, device, move_network=True):
    """ predict batchwise """
    # Build DataLoader
    if move_network:
        network = network.to(device)
    y = torch.Tensor(X.size()[0])
    data = DataLoader(TensorDataset(X, y), batch_size, False)
    # Batch prediction
    network.eval()

    with torch.no_grad():
        r, n = 0, X.size()[0]
        for batch_data in data:
            # Predict on batch
            X_batch = batch_data[0].to(device)
            y_batch_pred = network(X_batch).detach().cpu()
            # Infer prediction shape
            if r == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:])
            # Add to prediction tensor
            y_pred[r : min(n, r + batch_size)] = y_batch_pred
            r += batch_size
    return y_pred


def _eval_metrics(eval_specifics, hyperparameter_config, pipeline_config, training_components, X_train, Y_train, X_valid, Y_valid, log, epoch, budget):
    """ evaluate the metrics on specified datasets """

    training_components["network"].eval()

    # evalute on training set
    if eval_specifics["train"]:
        y_train_pred = predict(training_components["network"], X_train, hyperparameter_config["batch_size"], training_components["device"], move_network=False)
        for metric in training_components["metrics"]:
            log["train_" + metric.__name__] = metric(Y_train, y_train_pred)
    
    # evaluate on validation set
    if eval_specifics["valid"] and X_valid is not None and Y_valid is not None:
        y_val_pred = predict(training_components["network"], X_valid, hyperparameter_config["batch_size"], training_components["device"], move_network=False).to(training_components["device"])
        val_loss = training_components["loss_function"](Variable(y_val_pred), Variable(Y_valid.float()))

        log['val_loss'] = val_loss.data.item()
        if training_components["metrics"]:
            for metric in training_components["metrics"]:
                log["val_" + metric.__name__] = metric(Y_valid.float(), y_val_pred)

    # evaluate additional logs 
    if eval_specifics["logs"] and training_components["log_functions"]:
        for additional_log in training_components["log_functions"]:
            log[additional_log.__name__] = additional_log(training_components["network"], epoch)

    if 'use_tensorboard_logger' in pipeline_config and pipeline_config['use_tensorboard_logger']:
        import tensorboard_logger as tl
        worker_path = 'Train/'
        tl.log_value(worker_path + 'budget', float(budget), int(time.time()))
        for name, value in log.items():
            tl.log_value(worker_path + name, float(value), int(time.time()))