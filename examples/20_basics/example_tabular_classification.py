"""
======================
Tabular Classification
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

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.api.tabular_classification import TabularClassificationTask


############################################################################
# Data Loading
# ============
# X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
X, y = sklearn.datasets.fetch_openml(data_id=41167, return_X_y=True, as_frame=True)  # dionis
mem_limit = 6000
import numpy as np
import subprocess
subprocess.call("rm -r tmp", shell=True)

# mem_limit, N, D = 4000, 5500, 10000
# X = np.random.random((N, D)).astype(np.float32)
# y = (np.mean(X ** 2, axis=-1) >= 1. / 3.).astype(np.int32)[:, np.newaxis]

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
    temporary_directory='./tmp/autoPyTorch_example_tmp_01',
    output_directory='./tmp/autoPyTorch_example_out_01',
    # delete_tmp_folder_after_terminate=False,
    # delete_output_folder_after_terminate=False,
    seed=42,
)

############################################################################
# Search for an ensemble of machine learning algorithms
# =====================================================
# print(X_train.shape, X_train.dtype, f"{X_train.nbytes / 1e6:.3e} MB")
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test.copy(),
    y_test=y_test.copy(),
    dataset_name='Australian',
    optimize_metric='accuracy',
    min_budget=1,
    memory_limit=mem_limit,
    total_walltime_limit=60,
    func_eval_time_limit_secs=20,
    enable_traditional_pipeline=False,
    dataset_compression={
        "memory_allocation": 0.1,
        "methods": ["subsample"]
    }
)

############################################################################
# Print the final ensemble performance
# ====================================
y_pred = api.predict(X_test)
score = api.score(y_pred, y_test)
print(score)
# Print the final ensemble built by AutoPyTorch
print(api.show_models())

# Print statistics from search
print(api.sprint_statistics())
