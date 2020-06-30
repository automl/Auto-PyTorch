import argparse
import os as os
import numpy as np
import logging
import json
import random
import torch
import openml
from sklearn.model_selection import train_test_split
from IPython import embed

import ConfigSpace as cs
from autoPyTorch import HyperparameterSearchSpaceUpdates
from autoPyTorch.pipeline.nodes import LogFunctionsSelector, BaselineTrainer
from autoPyTorch import AutoNetClassification, AutoNetEnsemble
from autoPyTorch.pipeline.nodes import LogFunctionsSelector
from autoPyTorch.components.metrics.additional_logs import *
from autoPyTorch.utils.ensemble import test_predictions_for_ensemble

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def resplit(X, y, test_split=0.33):
        uniques, counts = np.unique(y, return_counts=True)
        indices = np.array(range(len(y)))
        if min(counts)>1:
            ind_train, ind_test = train_test_split(indices, stratify=y, test_size=test_split, shuffle=True, random_state=42)
        else:
            ind_train, ind_test = train_test_split(indices, test_size=test_split, shuffle=True, random_state=42)
        return ind_train, ind_test

def load_openml_data(openml_task_id):
    task = openml.tasks.get_task(task_id=openml_task_id)
    X, y = task.get_X_and_y()

    ten_splits = [3945, 146212, 34539, 168337, 168338, 7593, 189354, 168332, 168331, 168330, 168335]

    if openml_task_id in ten_splits:
        ind_train, ind_test = resplit(X, y)
    else:
        ind_train, ind_test = task.get_train_test_split_indices()

    return X[ind_train], X[ind_test], y[ind_train], y[ind_test]

def get_hyperparameter_search_space_updates_lcbench():
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
                                hyperparameter="shapedresnet:max_units",
                                value_range=[32,512],
                                log=True)
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:num_groups",
                                value_range=[1,5])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:blocks_per_group",
                                value_range=[1,3])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:resnet_shape",
                                value_range=["funnel"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:activation",
                                value_range=["relu"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:mlp_shape",
                                value_range=["funnel"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:num_layers",
                                value_range=[1, 6])
    return search_space_updates

def get_autonet_config_lcbench(min_budget, max_budget, max_runtime, run_id, task_id, num_workers, logdir, seed):
    autonet_config = {
            'additional_logs': [],
            'additional_metrics': ["balanced_accuracy"],
            'algorithm': 'bohb',
            'batch_loss_computation_techniques': ['standard', 'mixup'],
            'best_over_epochs': False,
            'budget_type': 'epochs',
            'categorical_features': None,
            #'cross_validator': 'stratified_k_fold',
            #'cross_validator_args': dict({"n_splits":5}),
            'cross_validator': 'none',
            'cuda': False,
            'dataset_name': None,
            'early_stopping_patience': 10,
            'early_stopping_reset_parameters': False,
            'embeddings': ['none', 'learned'],
            'eta': 2,
            'final_activation': 'softmax',
            'full_eval_each_epoch': True,
            'hyperparameter_search_space_updates': get_hyperparameter_search_space_updates_lcbench(),
            'imputation_strategies': ['mean'],
            'initialization_methods': ['default'],
            'initializer': 'simple_initializer',
            'log_level': 'info',
            'loss_modules': ['cross_entropy_weighted'],
            'lr_scheduler': ['cosine_annealing'],
            'max_budget': max_budget,
            'max_runtime': max_runtime,
            'memory_limit_mb': 12000,
            'min_budget': min_budget,
            'min_budget_for_cv': 0,
            'min_workers': num_workers,
            'network_interface_name': 'eth0',
            'networks': ['shapedmlpnet', 'shapedresnet'],
            'normalization_strategies': ['standardize'],
            'num_iterations': 300,
            'optimize_metric': 'accuracy',
            'optimizer': ['sgd', 'adam'],
            'over_sampling_methods': ['none'],
            'preprocessors': ['none', 'truncated_svd'],
            'random_seed': seed,
            'refit_validation_split': 0.33,
            'result_logger_dir': logdir,
            'run_id': run_id,
            'run_worker_on_master_node': True,
            'shuffle': True,
            'target_size_strategies': ['none'],
            'task_id': task_id,
            'torch_num_threads': 2,
            'under_sampling_methods': ['none'],
            'use_pynisher': True,
            'use_tensorboard_logger': False,
            'validation_split': 0.33,
            'working_dir': '.'
            }
    return autonet_config

def get_ensemble_config():
    ensemble_config = {
            "ensemble_size":50,
            "ensemble_only_consider_n_best":30,
            "ensemble_sorted_initialization_n_best":0
            }
    return ensemble_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--run_id", type=int, help="An id for the run.")
    parser.add_argument("--task_id", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--dataset_id", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ensemble_setting", type=str, choices=["normal", "ensemble"])
    parser.add_argument("--portfolio_type", type=str, choices=["none", "simple", "greedy"])
    parser.add_argument("--num_threads", type=str, default="1")
    parser.add_argument("--test", type=str, choices=["true", "false"], default="false")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = args.num_threads

    logdir = os.path.join("logs/", str(args.dataset_id), "run_"+str(args.run_id))

    # Get data
    openml_ids = [7593, 168331, 167200, 189905, 167152, 189860, 167190, 189871]
    openml_id = openml_ids[int(args.dataset_id)]
    seed_everything(42)
    X_train, X_test, y_train, y_test = load_openml_data(openml_id)

    # Seed
    seed = args.seed
    seed_everything(seed)

    # Get autonet config
    min_budget=10 if args.test=="false" else 1
    max_budget=50 if args.test=="false" else 4
    max_runtime = 10*60*60 if args.test=="false" else 30
    autonet_config = get_autonet_config_lcbench(min_budget=min_budget,
                                                max_budget=max_budget, 
                                                max_runtime=max_runtime,
                                                run_id=args.run_id, 
                                                task_id=args.task_id, 
                                                num_workers=args.num_workers, 
                                                logdir=logdir, 
                                                seed=args.seed)

    if args.portfolio_type=="none":
        autonet_config["algorithm"] = "bohb"
    else:
        autonet_config["algorithm"] = "portfolio_bohb"
        autonet_config["portfolio_type"] = args.portfolio_type

    # Categoricals
    cat_feats = [type(f)==str for f in X_train[0]]
    if any(cat_feats):
        autonet_config["categorical_features"] = cat_feats
    autonet_config["embeddings"] = ['none', 'learned']

    # Test logging
    autonet_config["additional_logs"] = [test_predictions_for_ensemble.__name__, test_result_ens.__name__]

    # Initialize (ensemble)
    if args.ensemble_setting == "ensemble":
        print("Using ensembles!")
        ensemble_config = get_ensemble_config()
        autonet_config = {**autonet_config, **ensemble_config}
        autonet = AutoNetEnsemble(AutoNetClassification, config_preset="full_cs", **autonet_config)
    elif args.ensemble_setting == "normal":
        autonet = AutoNetClassification(config_preset="full_cs", **autonet_config)

    # Test logging cont.
    autonet.pipeline[LogFunctionsSelector.get_name()].add_log_function(name=test_predictions_for_ensemble.__name__,
                                                                       log_function=test_predictions_for_ensemble(autonet, X_test, y_test),
                                                                       loss_transform=False)
    autonet.pipeline[LogFunctionsSelector.get_name()].add_log_function(name=test_result_ens.__name__,
                                                                       log_function=test_result_ens(autonet, X_test, y_test))

    autonet.pipeline[BaselineTrainer.get_name()].add_test_data(X_test)

    print(autonet.get_current_autonet_config())

    fit_results = autonet.fit(X_train, y_train, **autonet.get_current_autonet_config())
    
    score = autonet.score(X_test, y_test) if y_test is not None else None

    print("Test score:", score)

    # Write to json
    results = dict()
    results["run_id"] = int(args.run_id)
    results["test_score"] = score
    results["seed"] = int(seed)

    with open(logdir + "/results_dump.json", "w") as f:
        json.dump(results, f)
