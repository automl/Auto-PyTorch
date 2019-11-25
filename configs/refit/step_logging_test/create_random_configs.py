import os as os
import numpy as np
import random
import json

import ConfigSpace as cs
from autoPyTorch import AutoNetClassification
from autoPyTorch import HyperparameterSearchSpaceUpdates


def get_sampling_space():
    sampling_space = dict()
    sampling_space["batch_loss_computation_techniques"] = ['standard', 'mixup']
    sampling_space["networks"] = ['shapedmlpnet']
    sampling_space["optimizer"] = ['adam', 'sgd', 'adamw', 'rmsprop']
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


if __name__=="__main__":

    # Decrease hyperparameter range
    search_space_updates = HyperparameterSearchSpaceUpdates()
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:num_layers",
                                value_range=[1, 10])
    search_space_updates.append(node_name="LearningrateSchedulerSelector",
                                hyperparameter="cosine_annealing:T_max",
                                value_range=[50,50])

    sampling_space = get_sampling_space()
    autonet = AutoNetClassification(**sampling_space,
                                    hyperparameter_search_space_updates=search_space_updates)

    
    CS = autonet.get_hyperparameter_search_space()

    for cs_seed in range(1,100*39+1):

        np.random.seed(cs_seed)
        random.seed(cs_seed)
        CS.seed(cs_seed)
        hyperparameter_config = CS.sample_configuration().get_dictionary()

        for key in hyperparameter_config.keys():
            if hyperparameter_config[key]==True:
                hyperparameter_config[key]="True"
            elif hyperparameter_config[key]==False:
                hyperparameter_config[key]="False"

        combined_config = {**hyperparameter_config}
        
        with open("config_" + str(cs_seed) + ".json", "w") as f:
            json.dump(combined_config, f)
