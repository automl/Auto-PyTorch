__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
import logging

from autoPyTorch import AutoNetClassification, AutoNetMultilabel
import autoPyTorch.pipeline.nodes as autonet_nodes
from autoPyTorch.components.metrics.additional_logs import test_result
import autoPyTorch.components.metrics as autonet_metrics

from autoPyTorch.data_management.data_manager import DataManager

dm = DataManager(verbose=1)
dataset_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'datasets'))

# choose between the 5 defined testcases
TEST_CASE = 1

""" TEST CASE 1: Sparse data """
if TEST_CASE == 1:
    dm.read_data(os.path.join(dataset_dir, "automl/newsgroups/newsgroups_public.info"), is_classification=True)
    metric = "pac_metric"
    additional_metrices = ["accuracy"]

""" TEST CASE 2: Sparse binary data """
if TEST_CASE == 2:
    dm.read_data(os.path.join(dataset_dir, "automl/dorothea/dorothea_public.info"), is_classification=True)
    metric = "auc_metric"
    additional_metrices = ["accuracy"]

""" TEST CASE 3: Multilabel, sparse, binary, cv """
if TEST_CASE == 3:
    dm.read_data(os.path.join(dataset_dir, "automl/tania/tania_public.info"), is_classification=True)
    metric = "pac_metric"
    additional_metrices = []

""" TEST CASE 4: Openml, missing values """
if TEST_CASE == 4:
    dm.read_data("openml:188", is_classification=True)
    metric = "accuracy"
    additional_metrices = []

""" TEST CASE 5: MNIST """
if TEST_CASE == 5:
    dm.read_data("openml:40996", is_classification=True)
    metric = "accuracy"
    additional_metrices = []

# Generate autonet
autonet = AutoNetClassification() if TEST_CASE != 3 else AutoNetMultilabel()

# add metrics and test_result to pipeline
autonet.pipeline[autonet_nodes.LogFunctionsSelector.get_name()].add_log_function('test_result', test_result(autonet, dm.X_test, dm.Y_test), True)

# Fit autonet using train data
res = autonet.fit(min_budget=300,
                  max_budget=900, max_runtime=1800, budget_type='time',
                  normalization_strategies=['maxabs'],
                  optimize_metric=metric,
                  additional_metrics=additional_metrices,
                  cross_validator='stratified_k_fold',
                  cross_validator_args={'n_splits': 3},
                  preprocessors=["truncated_svd"],
                  log_level="debug",
                  X_train=dm.X_train,
                  Y_train=dm.Y_train,
                  X_valid=None,
                  Y_valid=None,
                  categorical_features=dm.categorical_features,
                  additional_logs=["test_result"])

# Calculate quality metrics using validation data.
autonet.score(dm.X_test, dm.Y_test)
print(res)
