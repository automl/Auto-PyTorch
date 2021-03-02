"""
======================
Time Series Classification on Sequential MNIST
======================

The following example shows how to fit a sample classification model
with AutoPyTorch
"""
import os
import tempfile as tmp
import warnings

import torch
from torch.utils.data import Subset

import torchvision

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

from autoPyTorch.api.time_series_classification import TimeSeriesClassificationTask
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


def get_search_space_updates():
    """
    Search space updates to the task can be added using HyperparameterSearchSpaceUpdates
    Returns:
        HyperparameterSearchSpaceUpdates
    """
    updates = HyperparameterSearchSpaceUpdates()
    updates.append(node_name="data_loader",
                   hyperparameter="batch_size",
                   value_range=[16, 512],
                   default_value=32)
    updates.append(node_name="lr_scheduler",
                   hyperparameter="CosineAnnealingLR:T_max",
                   value_range=[50, 60],
                   default_value=55)
    updates.append(node_name='optimizer',
                   hyperparameter='AdamOptimizer:lr',
                   value_range=[0.0001, 0.001],
                   default_value=0.0005)
    return updates


if __name__ == '__main__':
    ############################################################################
    # Data Loading
    # ============
    train_dataset = torchvision.datasets.MNIST(root=".", train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root=".", train=False)

    train_dataset = Subset(train_dataset, indices=torch.randperm(len(train_dataset))[:10000])
    test_dataset = Subset(train_dataset, indices=torch.randperm(len(test_dataset))[:100])

    X_train = np.empty((len(train_dataset), 28 * 28, 1), dtype=np.float32)
    y_train = np.empty(len(train_dataset), dtype=np.int32)
    X_test = np.empty((len(test_dataset), 28 * 28, 1), dtype=np.float32)
    y_test = np.empty(len(test_dataset), dtype=np.int32)

    for i, (image, label) in enumerate(train_dataset):
        X_train[i] = np.asarray(image).reshape(28 * 28, 1)
        y_train[i] = label

    for i, (image, label) in enumerate(test_dataset):
        X_test[i] = np.asarray(image).reshape(28 * 28, 1)
        y_test[i] = label

    ############################################################################
    # Build and fit a classifier
    # ==========================
    api = TimeSeriesClassificationTask(
        n_jobs=6,
        delete_tmp_folder_after_terminate=False,
        search_space_updates=get_search_space_updates(),
        exclude_components={"network_backbone": ["LSTMBackbone"]}
    )
    api.set_pipeline_config(device="cuda")
    api.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test.copy(),
        y_test=y_test.copy(),
        optimize_metric='accuracy',
        total_walltime_limit=1200,
        func_eval_time_limit=1200
    )

    ############################################################################
    # Print the final ensemble performance
    # ====================================
    print(api.run_history, api.trajectory)
    y_pred = api.predict(X_test)
    score = api.score(y_pred, y_test)
    print(score)
    print(api.show_models())
