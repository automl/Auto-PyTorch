import time
import os
import torch
import numpy as np

from torch.autograd import Variable
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper

# from util.transforms import mixup_data, mixup_criterion
# from checkpoints import save_checkpoint

class Trainer(object):
    def __init__(self, metrics, log_functions, loss_computation, model, criterion,
            budget, optimizer, training_techniques, logger, device, full_eval_each_epoch):
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.log_functions = log_functions
        self.model = model
        self.device = device

        for t in training_techniques:
            for key, value in t.training_components.items():
                setattr(self, key, value)
        self.training_techniques = training_techniques

        self.budget = budget
        self.loss_computation = loss_computation

        self.logger = logger
        self.fit_start_time = None

        self.eval_valid_each_epoch = full_eval_each_epoch or any(t.requires_eval_each_epoch() for t in self.training_techniques)
        self.eval_valid_on_snapshot = not self.eval_valid_each_epoch
        
        self.eval_additional_logs_each_epoch = full_eval_each_epoch and self.log_functions
        self.eval_additional_logs_on_snapshot = not full_eval_each_epoch and self.log_functions

        self.to(device)
    
    def prepare(self, pipeline_config, hyperparameter_config, fit_start_time):
        self.fit_start_time = fit_start_time
        self.loss_computation.set_up(
            pipeline_config=pipeline_config,
            hyperparameter_config=ConfigWrapper(hyperparameter_config["batch_loss_computation_technique"], hyperparameter_config),
            logger=self.logger)
        for t in self.training_techniques:
            t.set_up(trainer=self, pipeline_config=pipeline_config)
    
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
    
    @staticmethod
    def get_device(pipeline_config):
        if not torch.cuda.is_available():
            pipeline_config["cuda"] = False
        return torch.device('cuda:0' if pipeline_config['cuda'] else 'cpu')
    
    def on_epoch_start(self, log, epoch):
        for t in self.training_techniques:
            t.on_epoch_start(trainer=self, log=log, epoch=epoch)
    
    def on_epoch_end(self, log, epoch):
        return any([t.on_epoch_end(trainer=self, log=log, epoch=epoch) for t in self.training_techniques])
    
    def final_eval(self, opt_metric_name, logs, train_loader, valid_loader, best_over_epochs, refit):
        # select log
        if best_over_epochs:
            final_log = min(logs, key=lambda log: self.metrics[0].loss_transform(log[opt_metric_name]))
        else:
            final_log = None
            for t in self.training_techniques:
                log = t.select_log(trainer=self, logs=logs)
                if log:
                    final_log = log
            final_log = final_log or logs[-1]

        # validation on snapshot
        if self.eval_additional_logs_on_snapshot or self.eval_valid_on_snapshot or refit:
            self.model.load_snapshot()
            valid_metric_results = None
            if valid_loader is not None and self.eval_valid_on_snapshot:
                valid_metric_results = self.evaluate(valid_loader)

            for i, metric in enumerate(self.metrics):
                if valid_metric_results:
                    final_log['val_' + metric.name] = valid_metric_results[i]
            if self.eval_additional_logs_on_snapshot and not refit:
                    for additional_log in self.log_functions:
                        final_log[additional_log.name] = additional_log(self.model, None)
        return final_log

    def train(self, epoch, train_loader):
        '''
            Trains the model for a single epoch
        '''

        loss_sum = 0.0
        N = 0
        self.model.train()
        outputs_data = list()
        targets_data = list()

        for step, (data, targets) in enumerate(train_loader):
   
            # prepare
            data = data.to(self.device)
            targets = targets.to(self.device)

            data, criterion_kwargs = self.loss_computation.prepare_data(data, targets)
            data = Variable(data)
            batch_size = data.size(0)

            for t in self.training_techniques:
                t.on_batch_start(trainer=self, epoch=epoch, step=step, num_steps=len(train_loader))

            # training
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss_func = self.loss_computation.criterion(**criterion_kwargs)
            loss = loss_func(self.criterion, outputs)
            loss.backward()
            self.optimizer.step()

            # save for metric evaluation
            if self.model.final_activation is not None:
                outputs = self.model.final_activation(outputs)
            outputs_data.append(outputs.data.cpu().detach().numpy())
            targets_data.append(targets.data.cpu().detach().numpy())

            loss_sum += loss.item() * batch_size
            N += batch_size

            if any([t.on_batch_end(batch_loss=loss.item(), trainer=self, epoch=epoch, step=step, num_steps=len(train_loader))
                    for t in self.training_techniques]):
                return self.compute_metrics(outputs_data, targets_data), loss_sum / N, True
        return self.compute_metrics(outputs_data, targets_data), loss_sum / N, False


    def evaluate(self, test_loader):
        self.model.eval()

        outputs_data = list()
        targets_data = list()

        with torch.no_grad():
            for _, (data, targets) in enumerate(test_loader):
    
                data = data.to(self.device)
                data = Variable(data)
                outputs = self.model(data)

                outputs_data.append(outputs.data.cpu().detach().numpy())
                targets_data.append(targets.data.cpu().detach().numpy())

        self.model.train()
        return self.compute_metrics(outputs_data, targets_data)
    
    def compute_metrics(self, outputs_data, targets_data):
        outputs_data = np.vstack(outputs_data)
        targets_data = np.vstack(targets_data)
        return [metric(outputs_data, targets_data) for metric in self.metrics]
