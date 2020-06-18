import os, sys
# sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from autoPyTorch import AutoNetImageClassification
from autoPyTorch.data_management.data_manager import DataManager
import numpy as np
import datetime
if __name__=='__main__':
    path_to_cifar_csv = os.path.abspath("datasets/CIFAR10.csv")

    networks = ['resnet9', 'resnet']
    for network in networks:
        c = datetime.datetime.now()
        autonet = AutoNetImageClassification(config_preset="full_cs", 
                                            networks=[network], 
                                            lr_scheduler=['cosine_annealing'], 
                                            prefetch=False, 
                                            optimizer = ['sgd'],
                                            budget_type='epochs', 
                                            result_logger_dir="./results", 
                                            working_dir='./results/')
        config = autonet.get_current_autonet_config()
        hyperparameter_config = autonet.get_hyperparameter_search_space().sample_configuration().get_dictionary()
        if network == 'resnet9':
            hyperparameter_config['CreateImageDataLoader:batch_size'] = 120
            hyperparameter_config['NetworkSelectorDatasetInfo:resnet9:conv_bn'] = 'conv_pool_bn_act'
            hyperparameter_config['OptimizerSelector:optimizer:sgd:weight_decay'] = 5e-5*hyperparameter_config['CreateImageDataLoader:batch_size']
            hyperparameter_config['OptimizerSelector:optimizer:sgd:momentum'] = 0.65
            hyperparameter_config['OptimizerSelector:optimizer:sgd:learning_rate'] = 0.065


        result = autonet.refit(X_train=np.array([path_to_cifar_csv]),
                                        Y_train=np.array([0]),
                                        X_valid=None,
                                        Y_valid=None,
                                        hyperparameter_config=hyperparameter_config,
                                        autonet_config=config,
                                        budget=30)
        a = (datetime.datetime.now() -c).total_seconds()
        print(a, network)
        print(result)