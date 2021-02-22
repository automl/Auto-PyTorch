"""
======================
Time Series Classification
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

from sktime.datasets import load_gunpoint

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
    updates.append(node_name='network_backbone',
                   hyperparameter='ResNetBackbone:dropout',
                   value_range=[0, 0.5],
                   default_value=0.2)
    return updates


if __name__ == '__main__':
    ############################################################################
    # Data Loading
    # ============
    X, y = load_gunpoint(return_X_y=True)

    # Convert the pandas dataframes returned from load_gunpoint to 3D numpy array since that is
    # the format AutoPyTorch expects for now
    X = [X.iloc[i][0].values for i in range(len(X))]
    y = [int(y.iloc[i]) for i in range(len(y))]
    X = np.vstack(X)

    # Expand the last dimension because time series data has to be of shape [B, T, F]
    # where B is the batch size, T is the time dimension and F are the number of features per time step
    X = X[..., np.newaxis]

    # Subtract one from the labels because they are initially in {1, 2}, but are expected to be in {0, 1}
    y = np.array(y) - 1

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        random_state=1,
    )

    ############################################################################
    # Build and fit a classifier
    # ==========================
    api = TimeSeriesClassificationTask(
        delete_tmp_folder_after_terminate=False,
        search_space_updates=get_search_space_updates()
    )
    api.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test.copy(),
        y_test=y_test.copy(),
        optimize_metric='accuracy',
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
