__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption

import torch
import numpy as np

class MetricSelector(PipelineNode):
    def __init__(self):
        super(MetricSelector, self).__init__()

        self.metrics = dict()
        self.default_optimize_metric = None

    def fit(self, pipeline_config):
        optimize_metric = self.metrics[pipeline_config["optimize_metric"]]
        additional_metrics = [self.metrics[metric] for metric in pipeline_config["additional_metrics"] if metric != pipeline_config["optimize_metric"]]

        return {'optimize_metric': optimize_metric, 'additional_metrics': additional_metrics}

    def predict(self, optimize_metric):
        return { 'optimize_metric': optimize_metric }

    def add_metric(self, name, metric, loss_transform=False, 
                   requires_target_class_labels=False, is_default_optimize_metric=False):
        """Add a metric, this metric has to be a function that takes to arguments y_true and y_predict
        
        Arguments:
            name {string} -- name of metric for definition in config
            loss_transform {callable / boolean} -- transform metric value to minimizable loss. If True: loss = 1 - metric_value
            metric {function} -- metric function takes y_true and y_pred
            is_default_optimize_metric {bool} -- should the given metric be the default train metric if not specified in config
        """

        if (not hasattr(metric, '__call__')):
            raise ValueError("Metric has to be a function")

        ohe_transform = undo_ohe if requires_target_class_labels else no_transform
        if isinstance(loss_transform, bool):
            loss_transform = default_minimize_transform if loss_transform else no_transform

        self.metrics[name] = AutoNetMetric(name=name,
                                           metric=metric,
                                           loss_transform=loss_transform,
                                           ohe_transform=ohe_transform)

        if (not self.default_optimize_metric or is_default_optimize_metric):
            self.default_optimize_metric = name

    def remove_metric(self, name):
        del self.metrics[name]
        if (self.default_optimize_metric == name):
            if (len(self.metrics) > 0):
                self.default_optimize_metric = list(self.metrics.keys())[0]
            else:
                self.default_optimize_metric = None

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="optimize_metric", default=self.default_optimize_metric, type=str, choices=list(self.metrics.keys()),
                info="This is the meta train metric BOHB will try to optimize."),
            ConfigOption(name="additional_metrics", default=[], type=str, list=True, choices=list(self.metrics.keys()))
        ]
        return options


def default_minimize_transform(value):
    return 1 - value

def no_transform(value):
    return value

def ensure_numpy(y):
    if type(y)==torch.Tensor:
        return y.detach().cpu().numpy()
    return y

def undo_ohe(y):
    if len(y.shape) == 1:
        return(y)
    return np.argmax(y, axis=1)

class AutoNetMetric():
    def __init__(self, name, metric, loss_transform, ohe_transform):
        self.loss_transform = loss_transform
        self.metric = metric
        self.ohe_transform = ohe_transform
        self.name = name
    
    def __call__(self, Y_pred, Y_true):

        Y_pred = ensure_numpy(Y_pred)
        Y_true = ensure_numpy(Y_true)

        if len(Y_pred.shape) > len(Y_true.shape):
            Y_pred = undo_ohe(Y_pred)
        return self.metric(self.ohe_transform(Y_true), self.ohe_transform(Y_pred))

    def get_loss_value(self, Y_pred, Y_true):
        return self.loss_transform(self.__call__(Y_pred, Y_true))
