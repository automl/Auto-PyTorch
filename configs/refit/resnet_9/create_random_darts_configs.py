import numpy as np
import random
import json
from tqdm import tqdm

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from autoPyTorch import AutoNetImageClassification


def ensure_log_limits(par, min_par, max_par):
    log_par = np.log(par)
    return (log_par>min_par and log_par<max_par)


def get_sampling_space():
    sampling_space = dict()
    sampling_space["batch_loss_computation_techniques"] = ['mixup']
    sampling_space["networks"] = ['resnet9']
    sampling_space["optimizer"] = ['sgd']
    sampling_space["lr_scheduler"] = ['cosine_annealing']
    sampling_space["loss_modules"] = ['cross_entropy']
    return sampling_space


if __name__=="__main__":

    sampling_space = get_sampling_space()
    autonet = AutoNetImageClassification(config_preset="full_cs", **sampling_space)

    fixed_hyperpars = {
        "CreateImageDataLoader:batch_size": 96,
        "ImageAugmentation:augment": "True",
        "ImageAugmentation:cutout": "True",
        "ImageAugmentation:cutout_holes": 1,
        "ImageAugmentation:length" : 16,
        "ImageAugmentation:fastautoaugment": "False",
        "ImageAugmentation:autoaugment":"True",
        "LossModuleSelectorIndices:loss_module": "cross_entropy",
        "NetworkSelectorDatasetInfo:network": "resnet9",
        "OptimizerSelector:optimizer": "sgd",
        #"OptimizerSelector:sgd:learning_rate": 0.025,
        "OptimizerSelector:sgd:momentum": 0.9,
        #"OptimizerSelector:sgd:weight_decay": 0.0003,
        "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max": 50,
        "SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min": 1e-8,
        "SimpleLearningrateSchedulerSelector:lr_scheduler": "cosine_annealing",
        "SimpleTrainNode:batch_loss_computation_technique": "mixup",
        "SimpleTrainNode:mixup:alpha": 0.2,
        }

    cs = autonet.get_hyperparameter_search_space()

    # Set some hyperparameters as log-normal
    min_lr, max_lr, mean_lr = 1e-4, 1e-1, 0.025
    hp_lognormal_lr = CSH.NormalFloatHyperparameter('OptimizerSelector:sgd:learning_rate', mu=mean_lr, sigma=mean_lr*2, log=True)
    min_l2, max_l2, mean_l2 = 3e-5, 3e-1, 3e-4
    hp_lognormal_l2 = CSH.NormalFloatHyperparameter('OptimizerSelector:sgd:weight_decay', mu=mean_l2, sigma=mean_l2*2, log=True)
    del cs._hyperparameters["OptimizerSelector:sgd:learning_rate"]
    del cs._hyperparameters["OptimizerSelector:sgd:weight_decay"]
    cs.add_hyperparameters([hp_lognormal_lr, hp_lognormal_l2])

    # Sample configs
    start_at = 1
    n_configs = 100
    seed = start_at

    for i in tqdm(range(start_at,n_configs+1)):

        seed += 1
        cs.seed(seed)
        hyperparameter_config = cs.sample_configuration().get_dictionary()

        # ensure limits:
        while not ensure_log_limits(hyperparameter_config["OptimizerSelector:sgd:learning_rate"], min_lr, max_lr) or \
                  not ensure_log_limits(hyperparameter_config["OptimizerSelector:sgd:weight_decay"], min_l2, max_l2):
            seed +=1
            cs.seed(seed)
            hyperparameter_config = cs.sample_configuration().get_dictionary()

        hyperparameter_config["OptimizerSelector:sgd:learning_rate"] = np.log(hyperparameter_config["OptimizerSelector:sgd:learning_rate"])
        hyperparameter_config["OptimizerSelector:sgd:weight_decay"] = np.log(hyperparameter_config["OptimizerSelector:sgd:weight_decay"])

        # convert string to bool
        for key in hyperparameter_config.keys():
            if hyperparameter_config[key]==True:
                hyperparameter_config[key]="True"
            elif hyperparameter_config[key]==False:
                hyperparameter_config[key]="False"

        # set fixed parameters
        combined_config = {**hyperparameter_config, **fixed_hyperpars}
        
        # dump
        with open("config_" + str(i) + ".json", "w") as f:
            json.dump(combined_config, f)

    # dump cs
    from ConfigSpace.read_and_write import json as jason

    with open("configspace.json", "w") as f:
        f.write(jason.write(cs))

