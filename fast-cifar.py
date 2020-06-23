import os, sys
# sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from autoPyTorch import AutoNetImageClassification
from autoPyTorch.data_management.data_manager import DataManager
import numpy as np
import datetime
if __name__=='__main__':
    path_to_cifar_csv = os.path.abspath("datasets/CIFAR10.csv")
    
    networks = ['densenet', 'densenet-flexible', 'mobilenet', 'resnet152']
    prefetches = [True]#, False]
    for network in networks:
        for prefetch in prefetches:
            c = datetime.datetime.now()
            autonet = AutoNetImageClassification(config_preset="full_cs", 
                                                networks=[network], 
                                                lr_scheduler=['cosine_annealing'], 
                                                prefetch=prefetch, 
                                                optimizer = ['sgd'],
                                                budget_type='epochs', 
                                                result_logger_dir="./results", 
                                                working_dir='./results/')
            config = autonet.get_current_autonet_config()
            if network == 'resnet9':
                hyperparameter_config = {
                                            'NetworkSelectorDatasetInfo:network': 'resnet9',
                                            'OptimizerSelector:optimizer': 'sgd',
                                            'SimpleLearningrateSchedulerSelector:lr_scheduler': 'cosine_annealing',
                                            'CreateImageDataLoader:batch_size': 120,
                                            'ImageAugmentation:augment': False,
                                            'ImageAugmentation:cutout': False,
                                            'LossModuleSelectorIndices:loss_module': 'cross_entropy',
                                            'SimpleTrainNode:batch_loss_computation_technique': 'standard',
                                            'NetworkSelectorDatasetInfo:resnet9:conv_bn': 'conv_pool_bn_act',
                                            'OptimizerSelector:sgd:learning_rate': 0.065,
                                            'OptimizerSelector:sgd:momentum': 0.65,
                                            'OptimizerSelector:sgd:weight_decay': 0.006,
                                            'SimpleLearningrateSchedulerSelector:cosine_annealing:T_max': 228,
                                            'SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min': 2.9835939887187464e-08
                                        }

            elif network == 'resnet':
                hyperparameter_config = {
                                            "NetworkSelectorDatasetInfo:network":"resnet",
                                            "SimpleLearningrateSchedulerSelector:lr_scheduler":"cosine_annealing",
                                            'CreateImageDataLoader:batch_size': 120,
                                            "ImageAugmentation:augment":False,
                                            "ImageAugmentation:cutout":False,
                                            "LossModuleSelectorIndices:loss_module":"cross_entropy",
                                            'OptimizerSelector:optimizer': 'sgd',
                                            "SimpleTrainNode:batch_loss_computation_technique":"standard",
                                            "NetworkSelectorDatasetInfo:resnet:death_rate":0.5410896593583736,
                                            "NetworkSelectorDatasetInfo:resnet:initial_filters":13,
                                            "NetworkSelectorDatasetInfo:resnet:nr_main_blocks":1,
                                            "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_1":2,
                                            "NetworkSelectorDatasetInfo:resnet:res_branches_1":1,
                                            "NetworkSelectorDatasetInfo:resnet:widen_factor_1":0.8466406389676037,
                                            'OptimizerSelector:sgd:learning_rate': 0.065,
                                            'OptimizerSelector:sgd:momentum': 0.65,
                                            'OptimizerSelector:sgd:weight_decay': 0.006,
                                            "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max":34,
                                            "SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min":7.135867916289695e-08,
                                            }
            else:
                hyperparameter_config = autonet.get_hyperparameter_search_space().sample_configuration().get_dictionary()
            result = autonet.refit(X_train=np.array([path_to_cifar_csv]),
                                            Y_train=np.array([0]),
                                            X_valid=None,
                                            Y_valid=None,
                                            hyperparameter_config=hyperparameter_config,
                                            autonet_config=config,
                                            budget=1)
            print(autonet.get_pytorch_model())
            a = (datetime.datetime.now() -c).total_seconds()
            print(a, network, prefetch)
            print(result)