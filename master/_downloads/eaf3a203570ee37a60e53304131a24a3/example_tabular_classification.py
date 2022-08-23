"""
======================
Tabular Classification
======================

The following example shows how to fit a simple classification ensemble
with AutoPyTorch and refit the found ensemble.
"""
import os
import tempfile as tmp
import warnings

from autoPyTorch.datasets.resampling_strategy import CrossValTypes

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.api.tabular_classification import TabularClassificationTask


############################################################################
# Data Loading
# ============
X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
    random_state=1,
)

############################################################################
# Build and fit a classifier
# ==========================
api = TabularClassificationTask(
    # To maintain logs of the run, you can uncomment the
    # Following lines
    # temporary_directory='./tmp/autoPyTorch_example_tmp_01',
    # output_directory='./tmp/autoPyTorch_example_out_01',
    # delete_tmp_folder_after_terminate=False,
    # delete_output_folder_after_terminate=False,
    seed=42,
)

############################################################################
# Search for an ensemble of machine learning algorithms
# =====================================================
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test.copy(),
    y_test=y_test.copy(),
    dataset_name='Australian',
    optimize_metric='accuracy',
    total_walltime_limit=300,
    func_eval_time_limit_secs=50
)

############################################################################
# Print the final ensemble performance before refit
# =================================================

y_pred = api.predict(X_test)
score = api.score(y_pred, y_test)
print(score)

# Print statistics from search
print(api.sprint_statistics())

###########################################################################
# Refit the models on the full dataset.
# =====================================

api.refit(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    dataset_name="Australian",
    # you can change the resampling strategy to
    # for example, CrossValTypes.k_fold_cross_validation
    # to fit k fold models and have a voting classifier
    # resampling_strategy=CrossValTypes.k_fold_cross_validation
)

############################################################################
# Print the final ensemble performance after refit
# ================================================

y_pred = api.predict(X_test)
score = api.score(y_pred, y_test)
print(score)

# Print the final ensemble built by AutoPyTorch
print(api.show_models())
