import argparse
import numpy as np
import os as os
import openml
import json
import time
import random
import torch
from IPython import embed
from sklearn.model_selection import train_test_split

import ConfigSpace as cs
from autoPyTorch import AutoNetImageClassification
from autoPyTorch import HyperparameterSearchSpaceUpdates
from autoPyTorch.pipeline.nodes import LogFunctionsSelector
from autoPyTorch.pipeline.nodes.image.network_selector_datasetinfo import NetworkSelectorDatasetInfo
from autoPyTorch.components.metrics.additional_logs import *


def get_autonet_config():
    autonet_config = {
            "min_workers" : 1,
            "budget_type" : "epochs",
            "validation_split" : 0.2,
            "task_id" : 0,
            "use_tensorboard_logger" : False,
            "txt_logging":True,
            "optimize_metric" : "accuracy",
            "default_dataset_download_dir" : "./datasets/",
            "images_root_folders" : ["./datasets/"],
            "images_shape" : [3, 32, 32],
            "log_level" : "info",
            "dataloader_worker": 2,
            "loss_modules": ["cross_entropy"],
            "log_level":"info",
            "save_checkpoints":True
            }
    return autonet_config


def load_hyperparameter_config(config_dir):
    with open(config_dir, "r") as f:
        data = json.load(f)
    for key,val in data.items():
        if val=="True":
            data[key] = True
        elif val=="False":
            data[key] = False
    return data


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit a random config on an openml task')
    parser.add_argument("--run_id", type=int, help="An id for the run.")
    parser.add_argument("--offset", type=int, help="An offset to get larger array numbers than allowed by the cluster", default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--config_root_dir", type=str)
    parser.add_argument("--root_logdir", type=str)
    parser.add_argument("--dataloader_worker", type=int, default=2)
    parser.add_argument("--device_type", type=str, choices=["gpu", "cpu"])
    args = parser.parse_args()

    torch.backends.cudnn.benchmark=True

    # args
    run_id = args.run_id + args.offset
    seed = args.seed
    budget = args.budget
    root_logdir = args.root_logdir
    config_dir = args.config_root_dir
    device_type = args.device_type

    # Seed
    seed_everything(seed)

    # Settings
    logdir = os.path.join(root_logdir, "budget_"+str(budget) +  "_seed_"+str(seed) + "_run_" + str(run_id))
    hyperparameter_config_dir = os.path.join(config_dir, "config_"+ str(run_id) + ".json")

    # Sample config (autonet)
    autonet_config = get_autonet_config()
    autonet_config["random_seed"] = seed
    autonet_config["run_id"] = str(run_id)
    autonet_config["result_logger_dir"] = logdir
    autonet_config["dataloader_worker"] = args.dataloader_worker
    autonet_config["additional_logs"] = []#, test_cross_entropy.__name__, test_balanced_accuracy.__name__]

    if device_type == "gpu":
        autonet_config["cuda"] = True
    else:
        autonet_config["cuda"] = False

    # Initialize Autonet
    autonet = AutoNetImageClassification(**autonet_config)

    # Add additional logs
    gl = GradientLogger()
    lw_gl = LayerWiseGradientLogger()
    additional_logs = [gradient_max(gl), gradient_mean(gl), gradient_median(gl), gradient_std(gl),
                       gradient_q10(gl), gradient_q25(gl), gradient_q75(gl), gradient_q90(gl),
                       layer_wise_gradient_max(lw_gl), layer_wise_gradient_mean(lw_gl),
                       layer_wise_gradient_median(lw_gl), layer_wise_gradient_std(lw_gl),
                       layer_wise_gradient_q10(lw_gl), layer_wise_gradient_q25(lw_gl),
                       layer_wise_gradient_q75(lw_gl), layer_wise_gradient_q90(lw_gl),
                       gradient_norm()]

    for additional_log in additional_logs:
        autonet.pipeline[LogFunctionsSelector.get_name()].add_log_function(name=type(additional_log).__name__,
                                                                           log_function=additional_log)

        autonet_config["additional_logs"].append(type(additional_log).__name__)

    autonet_config["additional_logs"].append(test_result.__name__)

    autonet.pipeline[LogFunctionsSelector.get_name()].add_log_function(name=test_result.__name__, 
                                                                       log_function=test_result(autonet, np.array(["CIFAR10"]), None))
    #autonet.pipeline[LogFunctionsSelector.get_name()].add_log_function(name=test_cross_entropy.__name__,
    #                                                                   log_function=test_cross_entropy(autonet, X[ind_test], y[ind_test]))
    #autonet.pipeline[LogFunctionsSelector.get_name()].add_log_function(name=test_balanced_accuracy.__name__,
    #                                                                   log_function=test_balanced_accuracy(autonet, X[ind_test], y[ind_test]))
    
    # Load hyperparameters
    hyperparameter_config = load_hyperparameter_config(hyperparameter_config_dir)
    if "LearningrateSchedulerSelector:cosine_annealing:T_max" in hyperparameter_config.keys():
        hyperparameter_config["LearningrateSchedulerSelector:cosine_annealing:T_max"] = budget

    

    # Print Infos
    info = {
            "budget": budget,
            "seed" : int(seed),
            }

    print("Autonet config:", autonet.get_current_autonet_config())
    print("Hyperparameter config:", hyperparameter_config)
    print("Info", info)

    # Refit
    results = autonet.refit(X_train=np.array(["datasets/CIFAR10.csv"]), 
                            Y_train=np.array([0]),
                            X_valid=None,    # It will automatically split 0.2 (set in autonet config)
                            Y_valid=None,
                            hyperparameter_config=hyperparameter_config,
                            autonet_config=autonet.get_current_autonet_config(),
                            budget=budget)

    # Save model
    network = autonet.pipeline[NetworkSelectorDatasetInfo.get_name()].fit_output["network"]
    torch.save(network.state_dict(), logdir + "/model_state_dict.pt")

    # Write to json
    results["run_id"] = int(run_id)

    pop_keys = []
    for key in results["info"].keys():
        if key.startswith("layer"):
            pop_keys.append(key)

    for key in pop_keys:
        del results["info"][key]

    for k,v in results.items():
        if isinstance(v, (np.float32, np.float64)):
            results[k] = float(v)

    for k,v in results["info"].items():
        if isinstance(v, (np.float32, np.float64)):
            results["info"][k] = float(v)

    with open(logdir + "/results_dump.json", "w") as f:
        json.dump(results, f)

    with open(logdir + "/info.json", "w") as f:
        json.dump(info, f)
