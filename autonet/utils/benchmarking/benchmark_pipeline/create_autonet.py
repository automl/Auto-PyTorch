from autonet import AutoNetClassification, AutoNetRegression, AutoNetMultilabel
import autonet.pipeline.nodes as autonet_nodes
import autonet.components.metrics as autonet_metrics
from autonet.components.metrics.additional_logs import test_result

from autonet.utils.config.config_option import ConfigOption, to_bool
from autonet.pipeline.base.pipeline_node import PipelineNode

from autonet.data_management.data_manager import ProblemType

class CreateAutoNet(PipelineNode):

    def fit(self, data_manager):
        if (data_manager.problem_type == ProblemType.FeatureRegression):
            autonet = AutoNetRegression()
        elif (data_manager.problem_type == ProblemType.FeatureMultilabel):
            autonet = AutoNetMultilabel()
        elif (data_manager.problem_type == ProblemType.FeatureClassification):
            autonet = AutoNetClassification()
        else:
            raise ValueError('Problem type ' + str(data_manager.problem_type) + ' is not defined')

        autonet.pipeline[autonet_nodes.LogFunctionsSelector.get_name()].add_log_function(
            'test_result', test_result(autonet, data_manager.X_test, data_manager.Y_test))

        metrics = autonet.pipeline[autonet_nodes.MetricSelector.get_name()]
        metrics.add_metric('pac_metric', autonet_metrics.pac_metric)
        metrics.add_metric('balanced_accuracy', autonet_metrics.balanced_accuracy)
        metrics.add_metric('mean_distance', autonet_metrics.mean_distance)
        metrics.add_metric('multilabel_accuracy', autonet_metrics.multilabel_accuracy)
        metrics.add_metric('auc_metric', autonet_metrics.auc_metric)
        metrics.add_metric('accuracy', autonet_metrics.accuracy)

        return { 'autonet': autonet }