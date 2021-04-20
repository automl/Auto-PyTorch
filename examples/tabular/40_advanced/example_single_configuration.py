# -*- encoding: utf-8 -*-
"""
==========================
Fit a single configuration
==========================
*Auto-PyTorch* searches for the best combination of machine learning algorithms
and their hyper-parameter configuration for a given task.

This example shows how one can fit one of these pipelines, both, with a user defined
configuration, and a randomly sampled one form the configuration space.
The pipelines that Auto-PyTorch fits are compatible with Scikit-Learn API. You can
get further documentation about Scikit-Learn models here: <https://scikit-learn.org/stable/getting_started.html`>_
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
import sklearn.metrics

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes


if __name__ == '__main__':
    ############################################################################
    # Data Loading
    # ============

    X, y = sklearn.datasets.fetch_openml(data_id=3, return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.5, random_state=3
    )

    ############################################################################
    # Define an estimator
    # ============================

    # Search for a good configuration
    estimator = TabularClassificationTask(
        resampling_strategy=HoldoutValTypes.holdout_validation,
        resampling_strategy_args={'val_share': 0.33}
    )

    ###########################################################################
    # Fit an user provided configuration
    # ==================================

    # We will create a configuration that has a user defined
    # min_samples_split in the Random Forest. We recommend you to look into
    # how the ConfigSpace package works here:
    # https://automl.github.io/ConfigSpace/master/

    pipeline, run_info, run_value, dataset = estimator.fit_pipeline(X_train=X_train, y_train=y_train,
                                                                    dataset_name='kr-vs-kp',
                                                                    X_test=X_test, y_test=y_test,
                                                                    resampling_strategy=estimator.resampling_strategy,
                                                                    resampling_strategy_args=estimator.
                                                                    resampling_strategy_args,
                                                                    disable_file_output=False,
                                                                    )

    # This object complies with Scikit-Learn Pipeline API.
    # https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    print(pipeline.named_steps)

    # The fit_pipeline command also returns a named tuple with the pipeline constraints
    print(run_info)

    # The fit_pipeline command also returns a named tuple with train/test performance
    print(run_value)

    # We can make sure that our pipeline configuration was honored as follows
    print("Passed Configuration:", pipeline.config)
    print("Random Forest:", pipeline.named_steps['network'].choice.network)
