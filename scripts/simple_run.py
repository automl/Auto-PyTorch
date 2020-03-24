import os
import sys
import argparse
import numpy as np
import openml
import json
import time
import random
import torch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from autoPyTorch import AutoNetClassification
from autoPyTorch import HyperparameterSearchSpaceUpdates
from autoPyTorch.pipeline.nodes import LogFunctionsSelector
from autoPyTorch.components.metrics.additional_logs import *


def _get_shared_sampling_space():
    sampling_space = dict()
    sampling_space["batch_loss_computation_techniques"] = ['standard']
    sampling_space["optimizer"] = ['sgd']
    #sampling_space["preprocessors"] = ['truncated_svd', 'none']
    sampling_space["preprocessors"] = ['none']
    sampling_space["imputation_strategies"] = ['mean']
    sampling_space["lr_scheduler"] = ['cosine_annealing']
    sampling_space["normalization_strategies"] = ['standardize']
    sampling_space["over_sampling_methods"] = ['none']
    sampling_space["under_sampling_methods"] = ['none']
    sampling_space["target_size_strategies"] = ['none']
    sampling_space["embeddings"] = ['learned']
    sampling_space["initialization_methods"] = ['default']
    sampling_space["loss_modules"] = ['cross_entropy_weighted']
    return sampling_space


def get_mlpnet_configspace():
    sampling_space = _get_shared_sampling_space()
    sampling_space["networks"] = ['shapedmlpnet']

    search_space_updates = HyperparameterSearchSpaceUpdates()
    search_space_updates.append(node_name="InitializationSelector",
                                hyperparameter="initializer:initialize_bias",
                                value_range=["Yes"])
    search_space_updates.append(node_name="CreateDataLoader",
                                hyperparameter="batch_size",
                                value_range=[16, 512],
                                log=True)
    search_space_updates.append(node_name="LearningrateSchedulerSelector",
                                hyperparameter="cosine_annealing:T_max",
                                value_range=[50, 50])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:activation",
                                value_range=["relu"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:max_units",
                                value_range=[64, 1024],
                                log=True)
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:max_dropout",
                                value_range=[0.0,1.0])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:use_dropout",
                                value_range=[True])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:mlp_shape",
                                value_range=["funnel"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:num_layers",
                                value_range=[1, 5])

    autonet = AutoNetClassification(config_preset="full_cs",
                                    **sampling_space,
                                    hyperparameter_search_space_updates=search_space_updates)

    max_do = CSH.UniformFloatHyperparameter(name='NetworkSelector:shapedmlpnet:max_dropout', lower=0.0, upper=1.0)
    cs = autonet.get_hyperparameter_search_space()
    cs.add_hyperparameter(max_do)
    return cs


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit a random config on an openml task')
    parser.add_argument("--run_id", type=int, help="An id for the run.")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # Get data
    openml_task_ids = [3, 12, 31, 53, 3917, 7592, 9952, 9977, 9981, 10101, 14965, 146195, 146821, 146822, 
            #146825, # fashion mnist
            167119, 167120]

    # Get IDs with proper splits (ty matze) (the sleep is introduced to prevent a DOS attack on the openml server)
    new_task_ids = [] 
    for task_id in openml_task_ids:
        task = openml.tasks.get_task(task_id)
        try:
            time.sleep(args.run_id*0.01)
            tasks_with_same_dataset = openml.tasks.list_tasks( data_id=task.dataset_id, task_type_id=1, output_format='dataframe', ) 
            tasks_with_same_dataset = tasks_with_same_dataset.query("estimation_procedure == '33% Holdout set'" ) 
            if 'evaluation_measures' in tasks_with_same_dataset.columns: 
                tasks_with_same_dataset=tasks_with_same_dataset.query('evaluation_measures != evaluation_measures') 
            else:
                pass 
        except Exception:
            try:
                time.sleep(args.run_id*0.01)
                tasks_with_same_dataset = openml.tasks.list_tasks( data_id=task.dataset_id, task_type_id=1, output_format='dataframe', )
                tasks_with_same_dataset = tasks_with_same_dataset.query("estimation_procedure == '33% Holdout set'" )
                if 'evaluation_measures' in tasks_with_same_dataset.columns:
                    tasks_with_same_dataset=tasks_with_same_dataset.query('evaluation_measures != evaluation_measures')
                else:
                    pass
            except Exception:
                print(task)
                if len(tasks_with_same_dataset) > 1: 
                    raise ValueError(task) 
                elif len(tasks_with_same_dataset) == 0: 
                    raise ValueError(task)
        
        new_task_ids.append(tasks_with_same_dataset['tid'].iloc[0]) 

    openml_task_ids = new_task_ids
    openml_task_id = openml_task_ids[args.run_id % len(openml_task_ids)]
    task = openml.tasks.get_task(task_id=openml_task_id)
    X, y = task.get_X_and_y()
    ind_train, ind_test = task.get_train_test_split_indices()

    # Settings
    val_split = 0.33

    # arg dependent
    log_every_n_datapoints = None
    budget = 50
    logdir = "logs/simple_test/run_" + str(args.run_id) + "_" + str(openml_task_id)

    print("Using config", str(args.run_id//len(openml_task_ids)), "for task", openml_task_id, "and run id", args.run_id)

    # Seed
    seed = args.seed
    seed_everything(seed)
    
    # Sample config (autonet)
    sampling_space = dict()
    sampling_space["best_over_epochs"] = False
    sampling_space["random_seed"] = seed
    sampling_space["budget_type"] = "epochs"
    sampling_space["refit_validation_split"] = val_split
    sampling_space["validation_split"] = val_split
    sampling_space["log_level"] = "info"
    sampling_space["run_id"] = args.run_id
    sampling_space["task_id"] = 0
    sampling_space["use_tensorboard_logger"] = True
    sampling_space["result_logger_dir"] = logdir
    sampling_space["full_eval_each_epoch"] = True
    sampling_space["log_every_n_datapoints"] = log_every_n_datapoints
    sampling_space["optimize_metric"] = "accuracy"
    sampling_space["additional_metrics"] = ["cross_entropy", "balanced_accuracy"]
    sampling_space["additional_logs"] = [test_result.__name__, test_cross_entropy.__name__, test_balanced_accuracy.__name__]

    # Initialize Autonet
    autonet = AutoNetClassification(**sampling_space)

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
    autonet.pipeline[LogFunctionsSelector.get_name()].add_log_function(name=test_balanced_accuracy.__name__,
                                                                       log_function=test_balanced_accuracy(autonet, X[ind_test], y[ind_test]))


    # Sample hyperparameter config
    cs_mlpnet = get_mlpnet_configspace()
    hyperparameter_config = cs_mlpnet.sample_configuration().get_dictionary()
    hyperparameter_config["LearningrateSchedulerSelector:cosine_annealing:T_max"] = budget
    
    # Print Infos
    info = {
            "OpenML_task_id" : int(openml_task_id),
            "test_split" : len(y[ind_test])/(len(y[ind_train])+len(y[ind_test])),
            "budget": budget,
            "seed" : int(seed),
            "instances" : int(len(y[ind_train])+len(y[ind_test])),
            "classes": int(len(np.unique(y[ind_train]))),
            "features": int(X.shape[1])
            }

    print("Autonet config:", autonet.get_current_autonet_config())
    print("Hyperparameter config:", hyperparameter_config)
    print("Info", info)

    # Refit
    results = autonet.refit(X_train=X[ind_train],
                            Y_train=y[ind_train],
                            hyperparameter_config=hyperparameter_config,
                            autonet_config=autonet.get_current_autonet_config(),
                            budget=budget)

    # Write to json
    results["openml_task_id"] = int(openml_task_id)
    results["run_id"] = int(args.run_id)
    results["validation_split"] = sampling_space["refit_validation_split"]

    pop_keys = []
    for key in results["info"].keys():
        if key.startswith("layer"):
            pop_keys.append(key)

    for key in pop_keys:
        del results["info"][key]

    with open(logdir + "/results_dump.json", "w") as f:
        json.dump(results, f)

    with open(logdir + "/info.json", "w") as f:
        json.dump(info, f)
