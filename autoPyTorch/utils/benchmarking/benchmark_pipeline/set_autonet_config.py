
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

class SetAutoNetConfig(PipelineNode):

    def fit(self, pipeline_config, autonet, autonet_config_file, data_manager):
        parser = autonet.get_autonet_config_file_parser()
        config = parser.read(autonet_config_file)

        print("SetAutoNetConfig: additional logs ommited")
        #if ('additional_logs' not in config):
        #    config['additional_logs'] = ['test_result']

        if (pipeline_config['use_dataset_metric'] and data_manager.metric is not None):
            config['train_metric'] = data_manager.metric
        if (pipeline_config['use_dataset_max_runtime'] and data_manager.max_runtime is not None):
            config['max_runtime'] = data_manager.max_runtime

        if (pipeline_config['working_dir'] is not None):
            config['working_dir'] = pipeline_config['working_dir']
        if (pipeline_config['network_interface_name'] is not None):
            config['network_interface_name'] = pipeline_config['network_interface_name']

        config['log_level'] = pipeline_config['log_level']
        
        if data_manager.categorical_features:
            config['categorical_features'] = data_manager.categorical_features

        # if 'refit_config' in pipeline_config and pipeline_config['refit_config'] is not None:
        #     import json
        #     with open(pipeline_config['refit_config'], 'r') as f:
        #         refit_hyper_config = json.load(f)
        #     if 'incumbent_config_path' in refit_hyper_config:
        #         config['random_seed'] = refit_hyper_config['seed']
        #         config['dataset_order'] = refit_hyper_config['dataset_order']

        # Note: PrepareResultFolder will make a small run dependent update of the autonet_config
        autonet.update_autonet_config(**config)
        return dict()

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("use_dataset_metric", default=False, type=to_bool),
            ConfigOption("use_dataset_max_runtime", default=False, type=to_bool),
            ConfigOption("working_dir", default=None, type='directory'),
            ConfigOption("network_interface_name", default=None, type=str)
        ]
        return options
