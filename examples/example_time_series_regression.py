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

from sktime.datasets import load_italy_power_demand

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
                   value_range=[32, 64],
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
    # (Mostly copied from
    # https://github.com/sktime/sktime-dl/blob/master/examples/univariate_time_series_regression_and_forecasting.ipynb)
    # ============
    X_train_pd, _ = load_italy_power_demand(split='train', return_X_y=True)
    X_test_pd, _ = load_italy_power_demand(split='test', return_X_y=True)

    # Create some regression values.
    # Make the value y equal to the sum of the X values at time-steps 1 and 10.
    X_train = np.zeros((len(X_train_pd), 24, 1), dtype=float)
    y_train = np.zeros(len(X_train_pd), dtype=float)
    for i in range(len(X_train_pd)):
        y_train[i] = X_train_pd.iloc[i].iloc[0].iloc[1]
        y_train[i] = y_train[i] + X_train_pd.iloc[i].iloc[0].iloc[10]
        X_train[i] = X_train_pd.iloc[i].iloc[0][:, np.newaxis]

    X_test = np.zeros((len(X_test_pd), 24, 1), dtype=float)
    y_test = np.zeros(len(X_test_pd))
    for i in range(len(X_test_pd)):
        y_test[i] = X_test_pd.iloc[i].iloc[0].iloc[1]
        y_test[i] = y_test[i] + X_test_pd.iloc[i].iloc[0].iloc[10]
        X_test[i] = X_test_pd.iloc[i].iloc[0][:, np.newaxis]

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
        budget_type="runtime",
        budget=50,
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
