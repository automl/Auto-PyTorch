from autoPyTorch import AutoNetClassification, AutoNetRegression, AutoNetMultilabel, AutoNetEnsemble
from autoPyTorch.utils.ensemble import test_predictions_for_ensemble
import autoPyTorch.pipeline.nodes as autonet_nodes
import autoPyTorch.components.metrics as autonet_metrics
from autoPyTorch.components.metrics.additional_logs import test_result

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.data_management.data_manager import ProblemType

class CreateAutoNet(PipelineNode):

    def fit(self, pipeline_config, data_manager):
        if (data_manager.problem_type == ProblemType.FeatureRegression):
            autonet_type = AutoNetRegression
        elif (data_manager.problem_type == ProblemType.FeatureMultilabel):
            autonet_type = AutoNetMultilabel
        elif (data_manager.problem_type == ProblemType.FeatureClassification):
            autonet_type = AutoNetClassification
        else:
            raise ValueError('Problem type ' + str(data_manager.problem_type) + ' is not defined')

        autonet = autonet_type() if not pipeline_config["enable_ensemble"] else AutoNetEnsemble(autonet_type)
        test_logger = test_result if not pipeline_config["enable_ensemble"] else test_predictions_for_ensemble
        autonet.pipeline[autonet_nodes.LogFunctionsSelector.get_name()].add_log_function(
            name=test_logger.__name__, 
            log_function=test_logger(autonet, data_manager.X_test, data_manager.Y_test),
            loss_transform=(not pipeline_config["enable_ensemble"]))

        return { 'autonet': autonet }

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("enable_ensemble", default=False, type=to_bool)
        ]
        return options