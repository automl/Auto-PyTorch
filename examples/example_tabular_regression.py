"""
======================
Tabular Regression
======================

The following example shows how to fit a sample classification model
with AutoPyTorch
"""
import os
import tempfile as tmp
import typing
import warnings

from sklearn.datasets import make_regression

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn import model_selection, preprocessing

from autoPyTorch.api.tabular_regression import TabularRegressionTask
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


# Get the training data for tabular regression
def get_data_to_train() -> typing.Tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
    """
    This function returns a fit dictionary that within itself, contains all
    the information to fit a pipeline
    """

    # Get the training data for tabular regression
    # X, y = datasets.fetch_openml(name="cholesterol", return_X_y=True)

    # Use dummy data for now since there are problems with categorical columns
    X, y = make_regression(
        n_samples=5000,
        n_features=4,
        n_informative=3,
        n_targets=1,
        shuffle=True,
        random_state=0
    )

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        random_state=1,
    )

    return X_train, X_test, y_train, y_test


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
    X_train, X_test, y_train, y_test = get_data_to_train()

    # Scale the regression targets to have zero mean and unit variance.
    # This is important for Neural Networks since predicting large target values would require very large weights.
    # One can later rescale the network predictions like this: y_pred = y_pred_scaled * y_train_std + y_train_mean
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()

    y_train_scaled = (y_train - y_train_mean) / y_train_std
    y_test_scaled = (y_test - y_train_mean) / y_train_std

    datamanager = TabularDataset(
        X=X_train, Y=y_train_scaled,
        X_test=X_test, Y_test=y_test_scaled)

    ############################################################################
    # Build and fit a regressor
    # ==========================
    api = TabularRegressionTask(
        delete_tmp_folder_after_terminate=False,
        search_space_updates=get_search_space_updates()
    )

    api.set_pipeline_config(device="cuda")

    api.search(
        dataset=datamanager,
        optimize_metric='r2',
        traditional_per_total_budget=0,
        total_walltime_limit=500,
        func_eval_time_limit=150
    )

    ############################################################################
    # Print the final ensemble performance
    # ====================================
    print(api.run_history, api.trajectory)
    y_pred_scaled = api.predict(X_test)

    # Rescale the Neural Network predictions into the original target range
    y_pred = y_pred_scaled * y_train_std + y_train_mean
    score = api.score(y_pred, y_test)

    print(score)
