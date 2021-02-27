"""
======================
Time Series Regression
======================

The following example shows how to fit a sample classification model
with AutoPyTorch
"""
import os
import tempfile as tmp
import warnings

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import sklearn.model_selection

from autoPyTorch.api.time_series_regression import TimeSeriesRegressionTask
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

    # Create a dummy dataset consisting of sine and cosine waves
    length = 10
    sin_wave = np.sin(np.arange(length))
    cos_wave = np.cos(np.arange(length))
    sin_waves = []
    cos_waves = []
    # create a dummy dataset with 100 sin and 100 cosine waves
    for i in range(1000):
        # add some random noise so not every sample is equal
        sin_waves.append(sin_wave + np.random.randn(length) * 0.01)
        cos_waves.append(cos_wave + np.random.randn(length) * 0.01)
    sin_waves = np.stack(sin_waves)[..., np.newaxis]
    cos_waves = np.stack(cos_waves)[..., np.newaxis]

    X = np.concatenate([sin_waves, cos_waves])

    # use the last value of the time series as dummy regression target
    y = X[:, -1, 0]
    X = X[:, :-1, :]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        random_state=1
    )

    ############################################################################
    # Build and fit a regressor
    # ==========================
    api = TimeSeriesRegressionTask(
        delete_tmp_folder_after_terminate=False,
        search_space_updates=get_search_space_updates(),
        include_components={"network_backbone": ["InceptionTimeBackbone"]}
    )
    api.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test.copy(),
        y_test=y_test.copy(),
        optimize_metric='r2',
        total_walltime_limit=500,
        func_eval_time_limit=50
    )

    ############################################################################
    # Print the final ensemble performance
    # ====================================
    print(api.run_history, api.trajectory)
    y_pred = api.predict(X_test)
    score = api.score(y_pred, y_test)
    print(score)
