from autoPyTorch import AutoNetClassification, AutoNetRegression, AutoNetMultilabel, AutoNetImageClassification, AutoNetImageClassificationMultipleDatasets
import autoPyTorch.pipeline.nodes as autonet_nodes
import autoPyTorch.components.metrics as autonet_metrics
from autoPyTorch.components.metrics.additional_logs import test_result

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.data_management.data_manager import ProblemType

class CreateAutoNet(PipelineNode):

    def fit(self, data_manager):
        if (data_manager.problem_type == ProblemType.FeatureRegression):
            autonet = AutoNetRegression()
        elif (data_manager.problem_type == ProblemType.FeatureMultilabel):
            autonet = AutoNetMultilabel()
        elif (data_manager.problem_type == ProblemType.FeatureClassification):
            autonet = AutoNetClassification()
        elif data_manager.problem_type == ProblemType.ImageClassification:
            autonet = AutoNetImageClassification()
        elif data_manager.problem_type == ProblemType.ImageClassificationMultipleDatasets:
            autonet = AutoNetImageClassificationMultipleDatasets()
        else:
            raise ValueError('Problem type ' + str(data_manager.problem_type) + ' is not defined')

        print("Create Autonet: Logging on test data disabled during search (moved to X val for refit)")
        #autonet.pipeline[autonet_nodes.LogFunctionsSelector.get_name()].add_log_function(
        #    'test_result', test_result(autonet, data_manager.X_test, data_manager.Y_test))
        
        self.add_metrics(autonet)

        return { 'autonet': autonet }

    def add_metrics(self, autonet):
        metrics = autonet.pipeline[autonet_nodes.MetricSelector.get_name()]
        metrics.add_metric('pac_metric', autonet_metrics.pac_metric)
        metrics.add_metric('balanced_accuracy', autonet_metrics.balanced_accuracy)
        metrics.add_metric('mean_distance', autonet_metrics.mean_distance)
        metrics.add_metric('multilabel_accuracy', autonet_metrics.multilabel_accuracy)
        metrics.add_metric('auc_metric', autonet_metrics.auc_metric)
        metrics.add_metric('accuracy', autonet_metrics.accuracy)
        metrics.add_metric('top1', autonet_metrics.top1)
        metrics.add_metric('top3', autonet_metrics.top3)
        metrics.add_metric('top5', autonet_metrics.top5)
