import logging
import numpy as np
import os as os
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.data_management.data_manager import ImageManager
import json

class TransferAndRefit(PipelineNode):
    def fit(self, pipeline_config, autonet):
        if pipeline_config["transfer_dataset"] is not None:

            logging.getLogger('benchmark').info("Start refit on transfer dataset")

            # Get configs
            autonet_config = autonet.get_current_autonet_config()
            from autoPyTorch.utils.loggers import get_refit_config
            refit_config = get_refit_config(autonet_config['result_logger_dir'])

            directory = os.path.join(autonet_config['result_logger_dir'], 'transfer')

            autonet_config['result_logger_dir'] = directory
            autonet_config['save_checkpoints'] = False
            autonet_config['images_shape']=[3,64,64]
            autonet_config['additional_logs'] = []
            autonet_config['use_tensorboard_logger']=True
            autonet_config['result_logger_dir'] = directory
            pipeline_config['refit_config'] = refit_config

            # Get datasets
            transfer_dataset_train = pipeline_config["transfer_dataset"]
            transfer_dataset_test = pipeline_config["transfer_dataset_test"]

            # Set validation split
            if pipeline_config["transfer_val_split"] > 0.0:
                autonet_config["validation_split"] = pipeline_config["transfer_val_split"]

            # Score on test set if specified
            test_score = None
            if transfer_dataset_test is not None:
                dm_test = ImageManager(verbose=pipeline_config["data_manager_verbose"])
                dm_test.read_data(transfer_dataset_test,
                                  is_classification=(pipeline_config["problem_type"] in ['feature_classification', 'feature_multilabel', 'image_classification']),
                                  test_split=0.0)
                X_test, Y_test = dm_test.X_train, dm_test.Y_train.astype(np.int32)

            # Refit
            score, config = self.refit_autonet(
                pipeline_config, autonet, autonet_config,
                np.array([transfer_dataset_train]), np.array([0]),
                X_test, Y_test)

            # Create summary
            summary = dict()
            summary["val_score"] = score
            summary["incumbent_config"] = config
            summary["duration"] = pipeline_config['refit_budget'] or autonet_config['max_budget']

            # write as json
            with open(os.path.join(directory, "transfer_summary.json"), "w") as f:
                json.dump(summary, f)

        else:
            return dict()
        return dict()

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("transfer_dataset", default=None, type=str),
            ConfigOption("transfer_dataset_test", default=None, type=str),
            ConfigOption("transfer_val_split", default=0.0, type=float)
        ]
        return options

    def refit_autonet(self, pipeline_config, autonet, autonet_config, X_train, Y_train, X_valid, Y_valid):
        logging.getLogger('benchmark').debug("Refit autonet")
        
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        with open(pipeline_config['refit_config'], 'r') as f:
            refit_config = json.load(f)
        
        if 'incumbent_config_path' in refit_config:
            # > updates in set_autonet_config
            with open(refit_config['incumbent_config_path'], 'r') as f:
                config = json.load(f)
                autonet_config['random_seed'] = refit_config['seed']
                autonet_config['dataset_order'] = refit_config['dataset_order']
        else:
            config = refit_config

        result = autonet.refit(
            X_train, Y_train, 
            X_valid, Y_valid, 
            autonet_config=autonet_config,
            hyperparameter_config=config,
            budget=pipeline_config['refit_budget'] or autonet_config['max_budget'])

        score = result['final_metric_score']
        return score, config
