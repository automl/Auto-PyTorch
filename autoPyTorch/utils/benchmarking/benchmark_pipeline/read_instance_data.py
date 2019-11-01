import numpy as np
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.data_management.data_manager import DataManager, ImageManager

class ReadInstanceData(PipelineNode):

    def fit(self, pipeline_config, instance):
        # Get data manager for train, val, test data
        if pipeline_config['problem_type'] in ['feature_classification', 'feature_multilabel', 'feature_regression']:
            dm = DataManager(verbose=pipeline_config["data_manager_verbose"])
            if pipeline_config['test_instances'] is not None:
                dm_test = DataManager(verbose=pipeline_config["data_manager_verbose"])
        else:
            dm = ImageManager(verbose=pipeline_config["data_manager_verbose"])
            if pipeline_config['test_instances'] is not None:
                dm_test = ImageManager(verbose=pipeline_config["data_manager_verbose"])

        # Read data
        if pipeline_config['test_instances'] is not None:
            # Use given test set
            dm.read_data(instance,
                     is_classification=(pipeline_config["problem_type"] in ['feature_classification', 'feature_multilabel', 'image_classification']),
                     test_split=0.0)
            dm_test.read_data(pipeline_config['test_instances'],
                              is_classification=(pipeline_config["problem_type"] in ['feature_classification', 'feature_multilabel', 'image_classification']),
                              test_split=0.0)
            dm.X_test, dm.Y_test = dm_test.X_train, dm_test.Y_train.astype(np.int32)

        else:
            # Use test split
            dm.read_data(instance,
                is_classification=(pipeline_config["problem_type"] in ['feature_classification', 'feature_multilabel', 'image_classification']),
                test_split=pipeline_config["test_split"])

        return {"data_manager": dm}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("test_split", default=0.0, type=float),
            ConfigOption("problem_type", default='feature_classification', type=str, choices=['feature_classification', 'feature_multilabel', 'feature_regression', 'image_classification']),
            ConfigOption("data_manager_verbose", default=False, type=to_bool),
            ConfigOption("test_instances", default=None, type=str)
        ]
        return options
