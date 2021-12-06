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


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.fetch_openml(data_id=3, return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.5, random_state=3
)

############################################################################
# Define an estimator
# ===================

estimator = TabularClassificationTask(
    resampling_strategy=HoldoutValTypes.holdout_validation,
    resampling_strategy_args={'val_share': 0.33},
)

############################################################################
# Get a random configuration of the pipeline for current dataset
# ===============================================================

dataset = estimator.get_dataset(X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test,
                                dataset_name='kr-vs-kp')
configuration = estimator.get_search_space(dataset).sample_configuration()

print("Passed Configuration:", configuration)
###########################################################################
# Fit the configuration
# =====================

pipeline, run_info, run_value, dataset = estimator.fit_pipeline(dataset=dataset,
                                                                configuration=configuration,
                                                                budget_type='epochs',
                                                                budget=20,
                                                                run_time_limit_secs=100
                                                                )

# This object complies with Scikit-Learn Pipeline API.
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
print(pipeline.named_steps)

# The fit_pipeline command also returns a named tuple with the pipeline constraints
print(run_info)

# The fit_pipeline command also returns a named tuple with train/test performance
print(run_value)
