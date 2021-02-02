"""
======================
Tabular Classification
======================

The following example shows how to fit a sample classification model
with AutoPyTorch
"""
import typing
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


# Get the training data for tabular classification
def get_data_to_train() -> typing.Tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
    """
    This function returns a fit dictionary that within itself, contains all
    the information to fit a pipeline
    """

    # Get the training data for tabular classification
    # Move to Australian to showcase numerical vs categorical
    X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
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
    datamanager = TabularDataset(
        X=X_train, Y=y_train,
        X_test=X_test, Y_test=y_test)

    ############################################################################
    # Build and fit a classifier
    # ==========================
    api = TabularClassificationTask(
        delete_tmp_folder_after_terminate=False,
        search_space_updates=get_search_space_updates()
    )
    api.search(
        dataset=datamanager,
        optimize_metric='accuracy',
        total_walltime_limit=500,
        func_eval_time_limit=150
    )

    ############################################################################
    # Print the final ensemble performance
    # ====================================
    print(api.run_history, api.trajectory)
    y_pred = api.predict(X_test)
    score = api.score(y_pred, y_test)
    print(score)
