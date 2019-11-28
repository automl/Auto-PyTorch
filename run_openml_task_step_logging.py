import os
import sys
import argparse
import numpy as np
import openml
import json
import random
import torch

import ConfigSpace as cs
from autoPyTorch import AutoNetClassification
from autoPyTorch import HyperparameterSearchSpaceUpdates
from autoPyTorch.pipeline.nodes import LogFunctionsSelector
from autoPyTorch.components.metrics.additional_logs import *


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


def load_hyperparameter_config(config_id):
    config_dir = "configs/refit/step_logging_test/config_" + str(config_id) + ".json"
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
    args = parser.parse_args()


    # Sample OpenML task ID from AutoML Bench
    openml_task_ids = [3945, 14965, 7592, 9977, 7593, 146212, 167119, 146822, 146825, 34539, 146195, 146821, 146606, 146818, 168332, 
                       168331, 168330, 168337, 168335, 168338, 168329, 167120, 168868, 168908, 168912, 168910, 168909, 189354, 168911,
                       189356, 189355, 10101, 9952, 9981, 12, 3]
    openml_task_id = openml_task_ids[args.run_id % len(openml_task_ids)]

    # Seed
    seed = 1
    seed_everything(seed)

    # Get data
    task = openml.tasks.get_task(task_id=openml_task_id)
    X, y = task.get_X_and_y()
    ind_train, ind_test = task.get_train_test_split_indices()

    # APT settings
    budget = np.ceil(1e6 / len(y[ind_train])) # in epochs
    logdir = "logs/bench_results_step/run_" + str(args.run_id) + "_" + str(openml_task_id)
    
    # Sample config (autonet)
    sampling_space = get_sampling_space()
    sampling_space["best_over_epochs"] = False
    sampling_space["random_seed"] = 1
    sampling_space["budget_type"] = "epochs"
    sampling_space["refit_validation_split"] = 0.3
    sampling_space["log_level"] = "info"
    sampling_space["run_id"] = args.run_id
    sampling_space["task_id"] = 0
    sampling_space["use_tensorboard_logger"] = True
    sampling_space["result_logger_dir"] = logdir
    sampling_space["full_eval_each_epoch"] = True
    sampling_space["log_every_n_datapoints"] = 10000
    sampling_space["optimize_metric"] = "accuracy"
    sampling_space["additional_metrics"] = ["cross_entropy"]
    sampling_space["additional_logs"] = [test_result.__name__, test_cross_entropy.__name__]

    # Decrease hyperparameter range
    search_space_updates = HyperparameterSearchSpaceUpdates()
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:num_layers",
                                value_range=[1, 10])
    search_space_updates.append(node_name="LearningrateSchedulerSelector",
                                hyperparameter="cosine_annealing:T_max",
                                value_range=[budget, budget])

    # Initialize Autonet
    autonet = AutoNetClassification(**sampling_space,
                                    hyperparameter_search_space_updates=search_space_updates)


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

        sampling_space["additional_logs"].append(type(additional_log).__name__)

    autonet.pipeline[LogFunctionsSelector.get_name()].add_log_function(name=test_result.__name__, 
                                                                       log_function=test_result(autonet, X[ind_test], y[ind_test]))
    autonet.pipeline[LogFunctionsSelector.get_name()].add_log_function(name=test_cross_entropy.__name__,
                                                                       log_function=test_cross_entropy(autonet, X[ind_test], y[ind_test]))


    # Load hyperparameters
    hyperparameter_config = load_hyperparameter_config(args.run_id)
    if "LearningrateSchedulerSelector:cosine_annealing:T_max" in hyperparameter_config.keys():
        hyperparameter_config["LearningrateSchedulerSelector:cosine_annealing:T_max"] = budget

    # Print Infos
    print("Fitting on OpenML task", openml_task_id)
    print("Dataset points, features", X.shape)
    print("Train/test split", len(y[ind_train]), len(y[ind_test]))
    print("Fitting for epochs", budget)
    print("Sampling seed", seed)
    print("Autonet seed", sampling_space["random_seed"])
    print("Autonet config:", autonet.get_current_autonet_config())
    print("Hyperparameter config:", hyperparameter_config)

    # Refit
    results = autonet.refit(X_train=X[ind_train],
                            Y_train=y[ind_train],
                            hyperparameter_config=hyperparameter_config,
                            autonet_config=autonet.get_current_autonet_config(),
                            budget=budget)

    # Write to json
    results["openml_task_id"] = openml_task_id
    results["run_id"] = args.run_id
    results["validation_split"] = sampling_space["refit_validation_split"]

    pop_keys = []
    for key in results["info"].keys():
        if key.startswith("layer"):
            pop_keys.append(key)

    for key in pop_keys:
        del results["info"][key]

    with open(logdir + "/results_dump.json", "w") as file:
        json.dump(results, file)
