
from autonet.utils.config.config_option import ConfigOption, to_bool
from autonet.pipeline.base.pipeline_node import PipelineNode
from autonet.data_management.data_manager import DataManager

class ReadInstanceData(PipelineNode):

    def fit(self, pipeline_config, instance):
        assert pipeline_config['problem_type'] in ['feature_classification', 'feature_multilabel', 'feature_regression']
        dm = DataManager(verbose=pipeline_config["data_manager_verbose"])
        dm.read_data(instance,
            is_classification=(pipeline_config["problem_type"] in ['feature_classification', 'feature_multilabel']),
            test_split=pipeline_config["test_split"])
        return {"data_manager": dm}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("test_split", default=0.0, type=float),
            ConfigOption("problem_type", default='feature_classification', type=str, choices=['feature_classification', 'feature_multilabel', 'feature_regression']),
            ConfigOption("data_manager_verbose", default=False, type=to_bool),
        ]
        return options