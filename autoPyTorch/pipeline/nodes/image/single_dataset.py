import logging

from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser

class SingleDataset(SubPipelineNode):
    # Node for compatibility with MultipleDatasets model

    def __init__(self, sub_pipeline_nodes):
        super(SingleDataset, self).__init__(sub_pipeline_nodes)

        self.logger = logging.getLogger('autonet')


    def fit(self, hyperparameter_config, pipeline_config, X_train, Y_train, X_valid, Y_valid, budget, budget_type, config_id, working_directory):
        return self.sub_pipeline.fit_pipeline(hyperparameter_config=hyperparameter_config,
                                              pipeline_config=pipeline_config,
                                              X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid,
                                              budget=budget, budget_type=budget_type, config_id=config_id, working_directory=working_directory)


    def predict(self, pipeline_config, X):
        return self.sub_pipeline.predict_pipeline(pipeline_config=pipeline_config, X=X)

    def get_pipeline_config_options(self):
        options = [
            ConfigOption('dataset_order', default=None, type=int, list=True, info="Only used for multiple datasets."),

            #autonet.refit sets this to false to avoid refit budget issues
            ConfigOption('increase_number_of_trained_datasets', default=False, type=to_bool, info="Only used for multiple datasets.")
        ]
        return options
