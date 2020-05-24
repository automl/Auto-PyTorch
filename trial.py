from autoPyTorch import (AutoNetClassification,
                         AutoNetMultilabel,
                         AutoNetRegression,
                         AutoNetImageClassification,
                         AutoNetImageClassificationMultipleDatasets)

# Other imports for later usage
import pandas as pd
import numpy as np
import os as os
import openml
import json

autonet = AutoNetClassification(result_logger_dir="logs/")

current_configuration = autonet.get_current_autonet_config()

hyperparameter_search_space = autonet.get_hyperparameter_search_space()

print(current_configuration)
print(hyperparameter_search_space)

task = openml.tasks.get_task(task_id=31)
X, y = task.get_X_and_y()
ind_train, ind_test = task.get_train_test_split_indices()
X_train, Y_train = X[ind_train], y[ind_train]
X_test, Y_test = X[ind_test], y[ind_test]

results_fit = autonet.fit(X_train=X_train,
                          Y_train=Y_train,
                          validation_split=0.3,
                          max_runtime=600,
                          min_budget=30,
                          max_budget=100,
                          refit=True)

# Save fit results as json 
with open("logs/results_fit.json", "w") as file:
    json.dump(results_fit, file)
