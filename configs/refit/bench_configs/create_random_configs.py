import os as os
import numpy as np
import random
import json
from tqdm import tqdm

import ConfigSpace as CS
from autoPyTorch import AutoNetClassification
from autoPyTorch import HyperparameterSearchSpaceUpdates


def _get_shared_sampling_space():
    sampling_space = dict()
    sampling_space["batch_loss_computation_techniques"] = ['standard', 'mixup']
    sampling_space["optimizer"] = ['sgd']
    sampling_space["preprocessors"] = ['truncated_svd', 'none']
    sampling_space["imputation_strategies"] = ['mean']
    sampling_space["lr_scheduler"] = ['cosine_annealing', 'step']
    sampling_space["normalization_strategies"] = ['standardize']
    sampling_space["over_sampling_methods"] = ['none', 'random', 'smote']
    sampling_space["under_sampling_methods"] = ['none', 'random']
    sampling_space["target_size_strategies"] = ['upsample', 'downsample']
    sampling_space["embeddings"] = ['learned']
    sampling_space["initialization_methods"] = ['default', 'sparse']
    sampling_space["loss_modules"] = ['cross_entropy_weighted']
    return sampling_space


def get_mlpnet_configspace():
    sampling_space = _get_shared_sampling_space()
    sampling_space["networks"] = ['shapedmlpnet']

    search_space_updates = HyperparameterSearchSpaceUpdates()
    search_space_updates.append(node_name="CreateDataLoader",
                                hyperparameter="batch_size",
                                value_range=[32, 256],
                                log=True)
    search_space_updates.append(node_name="LearningrateSchedulerSelector",
                                hyperparameter="cosine_annealing:T_max",
                                value_range=[50, 50])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:max_units",
                                value_range=[64, 1024],
                                log=True)
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:num_layers",
                                value_range=[2, 8])
    
    autonet = AutoNetClassification(**sampling_space,
                                    hyperparameter_search_space_updates=search_space_updates)
    
    return autonet.get_hyperparameter_search_space()


def get_resnet_configspace():
    sampling_space = _get_shared_sampling_space()
    sampling_space["networks"] = ['shapedresnet']

    search_space_updates = HyperparameterSearchSpaceUpdates()
    search_space_updates.append(node_name="CreateDataLoader",
                                hyperparameter="batch_size",
                                value_range=[32, 256],
                                log=True)
    search_space_updates.append(node_name="LearningrateSchedulerSelector",
                                hyperparameter="cosine_annealing:T_max",
                                value_range=[50, 50])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:max_units",
                                value_range=[32,512],
                                log=True)
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:num_groups",
                                value_range=[1,5])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:blocks_per_group",
                                value_range=[1,3])

    autonet = AutoNetClassification(config_preset="full_cs",
                                    **sampling_space,
                                    hyperparameter_search_space_updates=search_space_updates)

    return autonet.get_hyperparameter_search_space()


def resnet_criterion(hyperpar_config):
    total_layers = hyperpar_config["NetworkSelector:shapedresnet:num_groups"] * hyperpar_config["NetworkSelector:shapedresnet:blocks_per_group"]
    return total_layers<=8


if __name__=="__main__":

    n_configs = 1000
    """
    
    # shaped mlp
    cs_mlpnet = get_mlpnet_configspace()
    seed=1
    for current_config in range(n_configs):

        np.random.seed(seed)
        random.seed(seed)
        cs_mlpnet.seed(seed)
        hyperparameter_config = cs_mlpnet.sample_configuration().get_dictionary()

        for key in hyperparameter_config.keys():
            if hyperparameter_config[key]==True:
                hyperparameter_config[key]="True"
            elif hyperparameter_config[key]==False:
                hyperparameter_config[key]="False"

        combined_config = {**hyperparameter_config}
        
        with open("shapedmlpnet/config_" + str(current_config) + ".json", "w") as f:
            json.dump(combined_config, f)

        seed += 1
    """


    # shaped resnet
    cs_resnet = get_resnet_configspace()
    seed = 1
    for current_config in range(n_configs):

        np.random.seed(seed)
        random.seed(seed)
        cs_resnet.seed(seed)
        hyperparameter_config = cs_resnet.sample_configuration().get_dictionary()

        while not resnet_criterion(hyperparameter_config):
            seed +=1
            np.random.seed(seed)
            random.seed(seed)
            cs_resnet.seed(seed)
            hyperparameter_config = cs_resnet.sample_configuration().get_dictionary()

        for key in hyperparameter_config.keys():
            if hyperparameter_config[key]==True:
                hyperparameter_config[key]="True"
            elif hyperparameter_config[key]==False:
                hyperparameter_config[key]="False"

        combined_config = {**hyperparameter_config}

        with open("shapedresnet/config_" + str(current_config) + ".json", "w") as f:
            json.dump(combined_config, f)

        seed +=1
