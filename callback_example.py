import os as os
import numpy as np
import json
import argparse
import pandas as pd
import torch
import random
import time
import openml
from sklearn.model_selection import train_test_split

from autoPyTorch import AutoNetClassification
from autoPyTorch.pipeline.nodes import LogFunctionsSelector
from autoPyTorch.components.metrics.additional_logs import *

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_33p_task(openml_task_id):
    """Searches openml tasks for the task on the same dataset but with 33 % test split"""
    task = openml.tasks.get_task(openml_task_id)
    tasks_with_same_dataset = openml.tasks.list_tasks( data_id=task.dataset_id, task_type_id=1, output_format='dataframe', )
    tasks_with_same_dataset = tasks_with_same_dataset.query("estimation_procedure == '33% Holdout set'" )
    if 'evaluation_measures' in tasks_with_same_dataset.columns:
        tasks_with_same_dataset=tasks_with_same_dataset.query('evaluation_measures != evaluation_measures')
    
    if len(tasks_with_same_dataset)>0:
        print("Found task with 33p split for %i" % openml_task_id)
        return tasks_with_same_dataset['tid'].iloc[0], True
    else:
        print("Could not find task with 33p split for %i" % openml_task_id)
        return openml_task_id, False

def resplit(X, y):
    seed_everything()
    test_split = 0.33
    indices = np.array(range(len(y)))
    ind_train, ind_test = train_test_split(indices, stratify=y, test_size=test_split, shuffle=True)
    return ind_train, ind_test

def load_data():
    openml_task_id = 31
    new_task_id, is_33p_split = get_33p_task(openml_task_id)
    task = openml.tasks.get_task(task_id=new_task_id)
    X, y = task.get_X_and_y()
    if is_33p_split:
        ind_train, ind_test = task.get_train_test_split_indices()
    else:
        ind_train, ind_test = resplit(X, y)
    return X[ind_train], X[ind_test], y[ind_train], y[ind_test]

def simple_callback(log):
    if log["val_accuracy"]>90:
        return True
    return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fit a random config on CIFAR task')
    parser.add_argument("--config_dir", type=str, default="./config_1.json")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="logs")
    args = parser.parse_args()

    logdir = args.logdir
    config_dir = args.config_dir

    # Set some autonet parameters, Add callbacks
    autonet_config = {
            "min_workers" : 1,
            "best_over_epochs" : False,
            "full_eval_each_epoch" : True,
            "budget_type" : "epochs",
            "optimize_metric" : "accuracy",
            "additional_metrics" : ["cross_entropy"],
            "validation_split" : 0.2,                          # Sets validation split
            "refit_validation_split" : 0.2,
            "use_tensorboard_logger" : True,
            "networks" : ['shapedmlpnet'],
            "log_level" : "info",
            "random_seed" : 1,
            "run_id" : args.run_id,
            "result_logger_dir": logdir,
            "random_seed":1,
            "callbacks": simple_callback,
            "additional_logs" : [test_result.__name__]}        # declare loggers

    # Initialize
    autonet = AutoNetClassification(**autonet_config)

    # Get data
    X_train, X_test, y_train, y_test = load_data()

    # Add logger
    autonet.pipeline[LogFunctionsSelector.get_name()].add_log_function(name=test_result.__name__,
                                                                       log_function=test_result(autonet, X_test, y_test))

    # Read hyperparameter config
    with open(config_dir, "r") as f:
        hyperparameter_config = json.load(f)

    # Fit a single config (=refit)
    autonet_config = autonet.get_current_autonet_config()
    result = autonet.refit(X_train=X_train, Y_train=y_train,
                           X_valid=None, Y_valid=None,
                           hyperparameter_config=hyperparameter_config,
                           autonet_config=autonet_config,
                           budget=50, budget_type="epochs")

    print("Done with refitting.")

    # Score
    score = autonet.score(X_test=X_test, Y_test=y_test)
    result["test_accuracy"] = score

    # Dump results
    with open(os.path.join(logdir, "final_output.json"), "w+") as f:
        json.dump(result, f)
