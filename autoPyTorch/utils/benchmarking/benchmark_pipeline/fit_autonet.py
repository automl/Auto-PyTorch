import time
import logging
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption
import json
import numpy as np

class FitAutoNet(PipelineNode):

    def __init__(self):
        super(FitAutoNet, self).__init__()

        # if we have the required module 'resource' (not available on windows!)
        self.guarantee_limits = module_exists("resource") and module_exists("pynisher")

    def fit(self, pipeline_config, autonet, data_manager, **kwargs):

        start_time = time.time()
        test_score = None

        if pipeline_config['refit_config'] is None:
            # Start search
            logging.getLogger('benchmark').debug("Fit autonet")

            # Email confirmation
            if pipeline_config['confirmation_gmail_user']:
                self.send_confirmation_mail(pipeline_config, autonet, data_manager)

            # Run fit
            fit_result = self.fit_autonet(autonet, data_manager)

            if pipeline_config['refit_budget'] is not None:
                # Refit
                import os
                import numpy as np
                autonet_config = autonet.get_current_autonet_config()
                from autoPyTorch.utils.loggers import get_refit_config
                refit_config = get_refit_config(autonet_config['result_logger_dir'])
                directory = os.path.join(autonet_config['result_logger_dir'], 'refit')

                autonet_config['result_logger_dir'] = directory
                autonet_config['save_checkpoints'] = False
                pipeline_config['refit_config'] = refit_config
                
                pipeline_config['refit_budget'] *= len(data_manager.X_train)
                job_id = max(autonet_config['task_id'], 1)
                if job_id == 1:
                    self.refit_autonet(
                        pipeline_config, autonet, autonet_config, 
                        data_manager.X_train, data_manager.Y_train, 
                        data_manager.X_valid, data_manager.Y_valid)

        else:
            # Refit
            autonet_config= autonet.get_current_autonet_config()
            fit_result = self.refit_autonet(
                pipeline_config, autonet, autonet_config, 
                data_manager.X_train, data_manager.Y_train, 
                data_manager.X_valid, data_manager.Y_valid)

        if data_manager.X_test is not None:
            # Score on test set
            import numpy as np
            test_score = autonet.score(data_manager.X_test, data_manager.Y_test.astype(np.int32))

        return { 'fit_duration': int(time.time() - start_time), 
                 'fit_result': fit_result,
                 'test_score': test_score}

    def fit_autonet(self, autonet, data_manager):
        return autonet.fit( data_manager.X_train, data_manager.Y_train, 
                            data_manager.X_valid, data_manager.Y_valid, 
                            refit=False)

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

        fit_result = autonet.refit(
            X_train, Y_train, 
            X_valid, Y_valid, 
            autonet_config=autonet_config,
            hyperparameter_config=config,
            budget=pipeline_config['refit_budget'] or autonet_config['max_budget'])

        logging.getLogger('benchmark').info("Result: " + str(fit_result))
        return fit_result

    def send_confirmation_mail(self, pipeline_config, autonet, data_manager):
        user = pipeline_config['confirmation_gmail_user']
        import pprint
        message = "\r\n".join(["Autonet run",
                               "Data:",
                               "%s",
                               "",
                               "Autonet Config:",
                               "%s"
                               "",
                               "",
                               "%s"]) % (pprint.pformat(data_manager.X_train.tolist()), pprint.pformat(autonet.get_current_autonet_config()), str(autonet.get_hyperparameter_search_space()))
        user = user + '+benchmark@gmail.com'
        from autoPyTorch.utils.mail import send_mail
        send_mail(user, 'Benchmark Start', message)
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption("refit_config", default=None, type='directory'),
            ConfigOption("refit_budget", default=None, type=int),
            ConfigOption("confirmation_gmail_user", default=None, type=str),
        ]
        return options


def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True
