"""
======================
Tabular Classification with Custom Configuration Space
======================

The following example shows how adjust the configuration space of
the search. Currently, there are two changes that can be made to the space:-
1. Adjust individual hyperparameters in the pipeline
2. Include or exclude components:
    a) include: Dictionary containing components to include. Key is the node
                name and Value is an Iterable of the names of the components
                to include. Only these components will be present in the
                search space.
    b) exclude: Dictionary containing components to exclude. Key is the node
                name and Value is an Iterable of the names of the components
                to exclude. All except these components will be present in
                the search space.
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

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.api.tabular_classification import TabularClassificationTask
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

############################################################################
# Data Loading
# ============
def get_data():
    X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        random_state=1,
    )
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()

    ############################################################################
    # Build and fit a classifier with include components
    # ==========================
    api = TabularClassificationTask(
        search_space_updates=get_search_space_updates(),
        include_components={'network_backbone': ['MLPBackbone'],
                            'encoder': ['OneHotEncoder']}
    )
    api.search(
        X_train=X_train.copy(),
        y_train=y_train.copy(),
        X_test=X_test.copy(),
        y_test=y_test.copy(),
        optimize_metric='accuracy',
        total_walltime_limit=300,
        func_eval_time_limit=50
    )

    ############################################################################
    # Print the final ensemble performance
    # ====================================
    print(api.run_history, api.trajectory)
    y_pred = api.predict(X_test)
    score = api.score(y_pred, y_test)
    print(score)
    print(api.show_models())

    ############################################################################
    # Build and fit a classifier with exclude components
    # ==========================
    api = TabularClassificationTask(
        search_space_updates=get_search_space_updates(),
        exclude_components={'network_backbone': ['MLPBackbone'],
                            'encoder': ['OneHotEncoder']}
    )
    api.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test.copy(),
        y_test=y_test.copy(),
        optimize_metric='accuracy',
        total_walltime_limit=300,
        func_eval_time_limit=50
    )

    ############################################################################
    # Print the final ensemble performance
    # ====================================
    print(api.run_history, api.trajectory)
    y_pred = api.predict(X_test)
    score = api.score(y_pred, y_test)
    print(score)
    print(api.show_models())
