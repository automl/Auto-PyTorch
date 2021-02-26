"""
Example file to be deleted
"""
import os

import sklearn.datasets

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
)
from test.utils import DisplayablePath



openml_id = 40981
resampling_strategy = CrossValTypes.k_fold_cross_validation
X, y = sklearn.datasets.fetch_openml(
    data_id=int(openml_id),
    return_X_y=True, as_frame=True
)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1)


if __name__ == '__main__':
    # Search for a good configuration
    estimator = TabularClassificationTask(
        temporary_directory='./tmp',
        delete_tmp_folder_after_terminate=False,
        resampling_strategy=resampling_strategy,
    )

    estimator.search(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        optimize_metric='accuracy',
        total_walltime_limit=150,
        func_eval_time_limit=50,
        traditional_per_total_budget=0
    )

    # Search for an existing run key in disc. A individual model might have
    # a timeout and hence was not written to disc
    for i, (run_key, value) in enumerate(estimator.run_history.data.items()):
        if i == 0:
            # Ignore dummy run
            continue
        if 'SUCCESS' not in str(value.status):
            continue

        run_key_model_run_dir = estimator._backend.get_numrun_directory(
            estimator.seed, run_key.config_id, run_key.budget)
        if os.path.exists(run_key_model_run_dir):
            break


    model_file = os.path.join(
        run_key_model_run_dir,
        f"{estimator.seed}.{run_key.config_id}.{run_key.budget}.cv_model"
    )

    paths = DisplayablePath.make_tree(os.path.dirname(run_key_model_run_dir))
    for path in paths:
        print(path.displayable())
