__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


from autonet.pipeline.base.pipeline_node import PipelineNode
from autonet.utils.config.config_option import ConfigOption

class MetricSelector(PipelineNode):
    def __init__(self):
        super(MetricSelector, self).__init__()

        self.metrics = dict()
        self.default_train_metric = None

    def fit(self, pipeline_config):
        train_metric = self.metrics[pipeline_config["train_metric"]]
        additional_metrics = [self.metrics[metric] for metric in pipeline_config["additional_metrics"] if metric != pipeline_config["train_metric"]]

        return {'train_metric': train_metric, 'additional_metrics': additional_metrics}

    def add_metric(self, name, metric, is_default_train_metric=False):
        """Add a metric, this metric has to be a function that takes to arguments y_true and y_predict
        
        Arguments:
            name {string} -- name of metric for definition in config
            metric {function} -- metric function takes y_true and y_pred
            is_default_train_metric {bool} -- should the given metric be the default train metric if not specified in config
        """

        if (not hasattr(metric, '__call__')):
            raise ValueError("Metric has to be a function")
        self.metrics[name] = metric
        metric.__name__ = name

        if (not self.default_train_metric or is_default_train_metric):
            self.default_train_metric = name

    def remove_metric(self, name):
        del self.metrics[name]
        if (self.default_train_metric == name):
            if (len(self.metrics) > 0):
                self.default_train_metric = list(self.metrics.keys())[0]
            else:
                self.default_train_metric = None

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="train_metric", default=self.default_train_metric, type=str, choices=list(self.metrics.keys()),
                info="This is the meta train metric BOHB will try to optimize."),
            ConfigOption(name="additional_metrics", default=[], type=str, list=True, choices=list(self.metrics.keys()))
        ]
        return options