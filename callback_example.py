import os as os
import numpy as np
import json
import argparse
import pandas as pd
import torch
import random
import openml

from autoPyTorch import AutoNetClassification


def load_data():
    task = openml.tasks.get_task(task_id=31)
    X, y = task.get_X_and_y()
    ind_train, ind_test = task.get_train_test_split_indices()
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
            "use_tensorboard_logger" : True,
            "networks" : ['shapedmlpnet'],
            "log_level" : "info",
            "random_seed" : 1,
            "run_id" : args.run_id,
            "result_logger_dir": logdir,
            "random_seed":1,
            "callbacks": simple_callback}

    # Initialize
    autonet = AutoNetClassification(**autonet_config)

    # Get data
    X_train, X_test, y_train, y_test = load_data()

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
